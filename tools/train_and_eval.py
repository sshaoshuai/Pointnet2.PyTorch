import _init_path
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import tensorboard_logger as tb_log
from dataset import KittiDataset
import argparse
import importlib

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--ckpt_save_interval", type=int, default=5)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--ckpt", type=str, default='None')

parser.add_argument("--net", type=str, default='pointnet2_msg')

parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--lr_decay', type=float, default=0.2)
parser.add_argument('--lr_clip', type=float, default=0.000001)
parser.add_argument('--decay_step_list', type=list, default=[50, 70, 80, 90])
parser.add_argument('--weight_decay', type=float, default=0.001)

parser.add_argument("--output_dir", type=str, default='output')
parser.add_argument("--extra_tag", type=str, default='default')

args = parser.parse_args()

FG_THRESH = 0.3


def log_print(info, log_f=None):
    print(info)
    if log_f is not None:
        print(info, file=log_f)


class DiceLoss(nn.Module):
    def __init__(self, ignore_target=-1):
        super().__init__()
        self.ignore_target = ignore_target

    def forward(self, input, target):
        """
        :param input: (N), logit
        :param target: (N), {0, 1}
        :return:
        """
        input = torch.sigmoid(input.view(-1))
        target = target.float().view(-1)
        mask = (target != self.ignore_target).float()
        return 1.0 - (torch.min(input, target) * mask).sum() / torch.clamp((torch.max(input, target) * mask).sum(), min=1.0)


def train_one_epoch(model, train_loader, optimizer, epoch, lr_scheduler, total_it, tb_log, log_f):
    model.train()
    log_print('===============TRAIN EPOCH %d================' % epoch, log_f=log_f)
    loss_func = DiceLoss(ignore_target=-1)

    for it, batch in enumerate(train_loader):
        optimizer.zero_grad()

        pts_input, cls_labels = batch['pts_input'], batch['cls_labels']
        pts_input = torch.from_numpy(pts_input).cuda(non_blocking=True).float()
        cls_labels = torch.from_numpy(cls_labels).cuda(non_blocking=True).long().view(-1)

        pred_cls = model(pts_input)
        pred_cls = pred_cls.view(-1)

        loss = loss_func(pred_cls, cls_labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_it += 1

        pred_class = (torch.sigmoid(pred_cls) > FG_THRESH)
        fg_mask = cls_labels > 0
        correct = ((pred_class.long() == cls_labels) & fg_mask).float().sum()
        union = fg_mask.sum().float() + (pred_class > 0).sum().float() - correct
        iou = correct / torch.clamp(union, min=1.0)

        cur_lr = lr_scheduler.get_lr()[0]
        tb_log.log_value('learning_rate', cur_lr, epoch)
        if tb_log is not None:
            tb_log.log_value('train_loss', loss, total_it)
            tb_log.log_value('train_fg_iou', iou, total_it)

        log_print('training epoch %d: it=%d/%d, total_it=%d, loss=%.5f, fg_iou=%.3f, lr=%f' %
                  (epoch, it, len(train_loader), total_it, loss.item(), iou.item(), cur_lr), log_f=log_f)

    return total_it


def eval_one_epoch(model, eval_loader, epoch, tb_log=None, log_f=None):
    model.train()
    log_print('===============EVAL EPOCH %d================' % epoch, log_f=log_f)

    iou_list = []
    for it, batch in enumerate(eval_loader):
        pts_input, cls_labels = batch['pts_input'], batch['cls_labels']
        pts_input = torch.from_numpy(pts_input).cuda(non_blocking=True).float()
        cls_labels = torch.from_numpy(cls_labels).cuda(non_blocking=True).long().view(-1)

        pred_cls = model(pts_input)
        pred_cls = pred_cls.view(-1)

        pred_class = (torch.sigmoid(pred_cls) > FG_THRESH)
        fg_mask = cls_labels > 0
        correct = ((pred_class.long() == cls_labels) & fg_mask).float().sum()
        union = fg_mask.sum().float() + (pred_class > 0).sum().float() - correct
        iou = correct / torch.clamp(union, min=1.0)

        iou_list.append(iou.item())
        log_print('EVAL: it=%d/%d, iou=%.3f' % (it, len(eval_loader), iou), log_f=log_f)

    iou_list = np.array(iou_list)
    avg_iou = iou_list.mean()
    if tb_log is not None:
        tb_log.log_value('eval_fg_iou', avg_iou, epoch)

    log_print('\nEpoch %d: Average IoU (samples=%d): %.6f' % (epoch, iou_list.__len__(), avg_iou), log_f=log_f)
    return avg_iou


def save_checkpoint(model, epoch, ckpt_name):
    if isinstance(model, torch.nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    state = {'epoch': epoch, 'model_state': model_state}
    ckpt_name = '{}.pth'.format(ckpt_name)
    torch.save(state, ckpt_name)


def load_checkpoint(model, filename):
    if os.path.isfile(filename):
        log_print("==> Loading from checkpoint %s" % filename)
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state'])
        log_print("==> Done")
    else:
        raise FileNotFoundError

    return epoch


def train_and_eval(model, train_loader, eval_loader, tb_log, ckpt_dir, log_f):
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in args.decay_step_list:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * args.lr_decay
        return max(cur_decay, args.lr_clip / args.lr)

    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd)

    total_it = 0
    for epoch in range(1, args.epochs + 1):
        lr_scheduler.step(epoch)
        total_it = train_one_epoch(model, train_loader, optimizer, epoch, lr_scheduler, total_it, tb_log, log_f)

        if epoch % args.ckpt_save_interval == 0:
            with torch.no_grad():
                avg_iou = eval_one_epoch(model, eval_loader, epoch, tb_log, log_f)
                ckpt_name = os.path.join(ckpt_dir, 'checkpoint_epoch_%d' % epoch)
                save_checkpoint(model, epoch, ckpt_name)


if __name__ == '__main__':
    MODEL = importlib.import_module(args.net)  # import network module
    model = MODEL.get_model(input_channels=0)

    eval_set = KittiDataset(root_dir='./data', mode='EVAL', split='val')
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                             num_workers=args.workers, collate_fn=eval_set.collate_batch)

    if args.mode == 'train':
        train_set = KittiDataset(root_dir='./data', mode='TRAIN', split='train')
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                  num_workers=args.workers, collate_fn=train_set.collate_batch)
        # output dir config
        output_dir = os.path.join(args.output_dir, args.extra_tag)
        os.makedirs(output_dir, exist_ok=True)
        tb_log.configure(os.path.join(output_dir, 'tensorboard'))
        ckpt_dir = os.path.join(output_dir, 'ckpt')
        os.makedirs(ckpt_dir, exist_ok=True)

        log_file = os.path.join(output_dir, 'log.txt')
        log_f = open(log_file, 'w')

        for key, val in vars(args).items():
            log_print("{:16} {}".format(key, val), log_f=log_f)

        # train and eval
        train_and_eval(model, train_loader, eval_loader, tb_log, ckpt_dir, log_f)
        log_f.close()
    elif args.mode == 'eval':
        epoch = load_checkpoint(model, args.ckpt)
        model.cuda()
        with torch.no_grad():
            avg_iou = eval_one_epoch(model, eval_loader, epoch)
    else:
        raise NotImplementedError


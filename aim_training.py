import argparse
import torch
import math
import sys
import time
import datetime
import torch.backends.cudnn as cudnn
import torch.utils.data
import model.att_in_mvs
import utils
import numpy as np
from timm.models import create_model
from load_data import dataset_path_iter, load_dataset, generate_data_path
from model.base_component import create_optimizer, NativeScalerWithGradNormCount
from torch import nn
from einops import rearrange
from pathlib import Path
from typing import Iterable


def get_args():
    parser = argparse.ArgumentParser('AIM training', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--save_ckpt_freq', default=20, type=int)

    # Model parameters
    parser.add_argument('--model', default='aim_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--multi_view', default=3, type=int)

    # Dataset parameters
    # parser.add_argument('--data_path', default='F:\\training_data\MVS\dtu_training', type=str, help='dataset path')
    parser.add_argument('--data_path', default='/home/gongshuai/pycharmProject/training_data/dtu_training', type=str, help='dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')

    # Optimizer parameters
    parser.add_argument('--opt', type=str, default='adamw', metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw")')
    parser.add_argument('--opt_eps', type=float, default=1e-8, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', type=float, default=None, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None,
                        help="""Final value of the weight decay. We use a cosine schedule for WD. 
                             (Set the same value with args.weight_decay to keep weight decay no change)""")

    # Learning rate and warmup LR
    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()


# Get model
def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(args.model, pretrained=False)
    return model


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, multi_views, start_steps=None, max_norm: float = 0.,
                    lr_scheduler=None, lr_schedule_values=None, wd_sechdule_values=None):
    model.train()
    loss_func = nn.MSELoss()

    # for step, batch in data_loader:
    for step, batch in enumerate(data_loader):
        # Assign learning rate & weight decay for each step
        batch = list(map(list, zip(*batch)))
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_sechdule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group['lr'] = lr_schedule_values[it] * param_group['lr_scale']
                if wd_sechdule_values is not None and param_group['weight_decay'] > 0:
                    param_group['weight_decay'] = wd_sechdule_values[it]

        # Load data using data path list
        images, depths = load_dataset(batch, multi_views)
        images = images.to(device, non_blocking=True)
        depths = depths.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            # Predict depth map
            predict_depth_map = model(images)
            # predict_depth_map = rearrange(predict_depth_map, 'b n p -> b (h p1) (w p2)',
            #                               p1=patch_size[0], p2=patch_size[1], h=224 / 16, w=224 / 16)
            loss = loss_func(input=predict_depth_map, target=depths)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # the attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()['scale']

        torch.cuda.synchronize()

        print(f'Epoch: {epoch}, batch: {step}, loss is {loss_value}')


def train(args):
    # Initialize distributed mode
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)
    cudnn.benchmark = True

    # Model
    model = get_model(args)
    patch_size = model.patch_size
    image_size = model.image_size
    print('Patch size = %s, Image size = %s' % (str(patch_size), str(image_size)))
    args.window_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
    args.patch_size = patch_size
    model.to(device)  # send model to each gpu
    model_without_ddp = model

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model = %s' % str(model))
    print('Number of params: {} M'.format(num_parameters / 1e6))
    # Compute used memory of model parameters
    mem_parameters = sum([np.prod(list(p.size())) for p in model.parameters()])

    # Dataset
    sample_path_list = generate_data_path(args.data_path, state='train', multi_views=args.multi_view)
    if True:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = len(sample_path_list) // args.batch_size // num_tasks
        sampler_train = torch.utils.data.DistributedSampler(sample_path_list,
                                                            num_replicas=num_tasks, rank=global_rank, shuffle=True)
        print('Sampler train = %s' % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(sample_path_list)
    data_loader_train = torch.utils.data.DataLoader(
        sample_path_list, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=utils.seed_worker)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    total_batch_size = args.batch_size * utils.get_world_size()
    args.lr = args.lr * total_batch_size / 256
    print('LR = %.8f' % args.lr)
    print('Batch size = %d' % total_batch_size)
    print('Number of training steps = %d' % num_training_steps_per_epoch)
    print('Number of training examples per epoch = %d' % (total_batch_size * num_training_steps_per_epoch))

    # Loss
    loss_scaler = NativeScalerWithGradNormCount()
    # Optimizers
    optimizer = create_optimizer(args, model_without_ddp)

    # Learning rate and weight decay schedule
    print('Use step level LR & WD scheduler!')
    lr_schedule_values = utils.cosine_scheduler(args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
                                                warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps)
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)

    utils.auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler)
    print(f'Start training for {args.epochs} epochs')

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler, args.multi_view
                        , start_steps=epoch * num_training_steps_per_epoch, max_norm=args.clip_grad
                        , lr_schedule_values=lr_schedule_values, wd_sechdule_values=wd_schedule_values)

        if args.output_dir and utils.is_main_process():
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(args=args, epoch=epoch, model=model, optimizer=optimizer, loss_scaler=loss_scaler)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    train(opts)

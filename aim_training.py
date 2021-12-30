import argparse
import torch
import math
import sys
import time
import datetime
import model.att_in_mvs
from timm.models import create_model
from load_data import dataset_path_iter, load_dataset, generate_data_path
from model.base_component import create_optimizer, auto_load_model, save_model
from torch import nn
from einops import rearrange
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser('AIM training', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)
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
    parser.add_argument('--imagenet_default_mean_and std', default=True, action='store_true')

    # Optimizer parameters
    parser.add_argument('--opt', type=str, default='adamw', metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw")')
    parser.add_argument('--opt_eps', type=float, default=1e-8, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', type=float, default=None, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    return parser.parse_args()


# Get model
def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(args.model, pretrained=False)
    return model


def train_one_epoch(model, dataset_path, batch_size, patch_size, optimizer, loss_func, epoch, multi_views):
    model.train()

    for i, data_path in dataset_path_iter(dataset_path, batch_size):
        # Load data using data path list
        images, depths = load_dataset(data_path, multi_views)

        # calculate the predict label
        # images_squeeze = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size[0],
        #                            p2=patch_size[1])
        # images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)) / \
        #               (images_squeeze.var(dim=-2, unbiased=True, keepdim=True) + 1e-6)
        # images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
        #
        # predict_depth_map = model(images_patch)
        predict_depth_map = model(images)
        # predict_depth_map = rearrange(predict_depth_map, 'b n p -> b (h p1) (w p2)',
        #                               p1=patch_size[0], p2=patch_size[1], h=224 / 16, w=224 / 16)
        loss = loss_func(input=predict_depth_map, target=depths)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch}, batch: {i}, loss is {loss_value}')


def train(args):
    # Model
    model = get_model(args)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model = %s' % str(model))
    print('Number of params: {} M'.format(num_parameters / 1e6))
    # Dataset
    sample_path_list = generate_data_path(args.data_path, state='train', multi_views=args.multi_view)
    # Loss
    loss_func = nn.MSELoss()
    # Optimizer
    optimizer = create_optimizer(args, model)

    auto_load_model(args, model)
    print(f'Start training for {args.epochs} epochs')
    start_time = time.time()

    num_epochs = args.epochs
    batch_size = args.batch_size
    for epoch in range(args.start_epoch, num_epochs):
        train_one_epoch(model, sample_path_list, batch_size, (16, 16), optimizer, loss_func, epoch, args.multi_view)

        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                save_model(args=args, epoch=epoch, model=model, optimizer=optimizer)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    train(opts)
    # test()
    # x = torch.randn((2, 1280, 768))
    # sum_func = torch.sum
    # x = sum_func(x, dim=1)
    # print('x = ', x)

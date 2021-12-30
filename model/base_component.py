import torch
import numpy as np
import os
import io
from torch import optim as optim
from pathlib import Path
from timm.utils import get_state_dict


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    """ Sinusoid position encoding table """
    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def create_optimizer(args, model):
    parameters = model.parameters()
    opt_args = dict(lr=args.lr)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['opt_betas'] = args.opt_betas
    print('optimizer setting: ', opt_args)
    opt_args.pop('eps', None)
    optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)

    return optimizer


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    :param model_ema:
    :param checkpoint:
    :return:
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def auto_load_model(args, model, model_ema=None):
    output_dir = args.output_dir
    # Deepspeed, only support '--auto_resume'
    if args.auto_resume:
        import glob
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
            print('Auto resume checkpoint: %d' % latest_ckpt)
            _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
            args.start_epoch = client_states['epoch'] + 1
            if model_ema is not None:
                if args.model_ema:
                    _load_checkpoint_for_ema(model_ema, client_states['model_ema'])


def is_main_process():
    # return get_rank() == 0
    return True


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def save_model(args, epoch, model, optimizer, loss_scaler=None, model_ema=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checlpoint_path in checkpoint_paths:
            to_save = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args
            }

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            save_on_master(to_save, checlpoint_path)
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=args.output_dir, tag='checkpoint-%s' % epoch_name, client_state=client_state)


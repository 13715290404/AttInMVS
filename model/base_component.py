import torch
import numpy as np
import json
from torch import optim as optim
from torch._six import inf

from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.novograd import NovoGrad
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP


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


try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD

    has_apex = True
except ImportError:
    has_apex = False


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_layer_id=None, get_layer_scale=None):
    """
    Get parameter groups, including no_decay and decay group
    :param model: model
    :param weight_decay: weight decay
    :param skip_list: the list of no weight decay layers' name
    :param get_layer_id: the function to get layer id by layer name
    :param get_layer_scale: the function to get layer learning rate scale by layer id
    :return: groups which layers are classified to no_decay or decay group
    """
    parameter_group_names = {}
    parameter_group_vars = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith('.bias') or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_layer_id is not None:
            layer_id = get_layer_id(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.
            parameter_group_names[group_name] = {
                'weight_decay': this_weight_decay,
                'params': [],
                'lr_scale': scale
            }
            parameter_group_vars[group_name] = {
                'weight_decay': this_weight_decay,
                'params': [],
                'lr_scale': scale
            }

        parameter_group_vars[group_name]['params'].append(param)
        parameter_group_names[group_name]['params'].append(name)
    print('Param groups = %s' % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(args, model, filter_bias_and_bn=True, skip_list=None, get_layer_id=None, get_layer_scale=None):
    """
    Create optimizer
    :param args: args
    :param model: model
    :param filter_bias_and_bn: True/False
    :param skip_list: the list of no weight decay layers' name
    :param get_layer_id: the function to get layer id by layer name
    :param get_layer_scale: the function to get layer learning rate scale by layer id
    :return: optimizer
    """
    opt = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(model, weight_decay, skip, get_layer_id, get_layer_scale)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    # Optimizer args: lr, weight_decay, eps, opt_betas
    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['opt_betas'] = args.opt_betas
    print('optimizer setting: ', opt_args)

    opts = opt.split('_')
    opt = opts[-1]
    if opt == 'sgd' or opt == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt == 'sgdp':
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt == 'adadelta':
        optimizer = optim.adadelta(parameters, **opt_args)
    elif opt == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt == 'novograd':
        optimizer = NovoGrad(parameters, **opt_args)
    elif opt == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and 'Invalid optimizer'
        raise ValueError

    if len(opts) > 1:
        if opt[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer


def get_grad_norm_(parameters, norm_type: float = 2.) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                                norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    """
    Scale loss and compute grad, as well as update grad in cuda.amp.GradScaler()
    AMP: automatic mixed precision - 自动混合精度, torch.FloatTensor and torch.HalfTensor in CUDA
    Loss scale: magnify loss to prevent grad underflow,
                and the grad will be unscaled when update params in backward
    """
    state_dict_key = 'amp_scaler'

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

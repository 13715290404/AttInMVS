import numpy as np
from torch import nn
from timm.models import create_model


def model_size(model, inputs, type_size=4):
    """
    Compute the number of model parameters and the size of output in each layer for given input
    :param model: model
    :param inputs: input
    :param type_size: data type, default 4 is float32
    :return:
    """
    parameters = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {}: params: {:4f}M'.format(model._get_name(), parameters * type_size / 1000 / 1000))

    inputs_ = inputs.clone()
    inputs_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(inputs_)
        out_sizes.append(np.array(out.size()))
        inputs_ = out

    total_num = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_num += nums

    print('Model {}: intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_num * type_size / 1000 / 1000))
    print('Model {}: intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_num * type_size * 2 / 1000 / 1000))

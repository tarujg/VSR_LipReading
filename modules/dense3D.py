

"""
Authors: Utkrisht Rajkumar, Subrato Chakravorty, Taruj Goyal, Kaustav Datta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, dropout_p):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.dropout_p = dropout_p

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.dropout_p > 0:
            new_features = F.dropout(new_features, p=self.dropout_p, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, dropout_p):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, dropout_p)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))


class Dense3D(torch.nn.Module):
    def __init__(self, growth_rate=8, num_init_features=32, bn_size=4, dropout_p=0):
        super(Dense3D, self).__init__()
        block_config = (4, 4)

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(3, num_init_features, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))),
        ]))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):

            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, dropout_p=dropout_p)
            self.features.add_module('denseblock%d' % (i + 1), block)

            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features)
                self.features.add_module('transition%d' % (i + 1), trans)

        self.features.add_module('norm%d' % (len(block_config)), nn.BatchNorm3d(num_features))
        self.features.add_module('pool', nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))

    def forward(self, x):
        return self.features(x)


if __name__ == '__main__':
    (B, C, T, H, W) = (16, 3, 75, 64, 128)
    data = torch.zeros((B, C, T, H, W))
    net = Dense3D('')
    # for k, v in m.state_dict().items():
    #     print(k)
    #print(net)
    print(net(data).size())

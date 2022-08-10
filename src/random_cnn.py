import random

import torch
from torch import nn


def get_conv_out_dim(conv_layer, input_dim):
    return (input_dim - torch.tensor(conv_layer.kernel_size) + 2 * torch.tensor(
        conv_layer.padding)) // torch.tensor(conv_layer.stride) + 1


def get_pooling_out_dim(pooling_layer, input_dim):
    return (input_dim + 2 * torch.tensor(pooling_layer.padding) - torch.tensor(
        pooling_layer.kernel_size)) // torch.tensor(pooling_layer.stride) + 1


def generate_pooling_layer(config):
    pooling_type = random.choice(config['generation']['pooling']['type'])
    pooling_kernel_size = random.choice(
        config['generation']['pooling']['kernel_size']
    )
    pooling_stride = random.choice(config['generation']['pooling']['stride'])
    pooling_padding = random.choice(config['generation']['pooling']['padding'])

    if pooling_type == 'Max':
        return nn.MaxPool2d(
            kernel_size=pooling_kernel_size,
            stride=pooling_stride,
            padding=pooling_padding,
        )
    return nn.AvgPool2d(
        kernel_size=pooling_kernel_size,
        stride=pooling_stride,
        padding=pooling_padding
    )


def generate_conv_layer(in_channels, out_channels, config):
    kernel_size = random.choice(config['generation']['conv']['kernel_size'])
    stride = random.choice(config['generation']['conv']['stride'])
    padding = random.choice(config['generation']['conv']['padding'])
    layer = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    return layer


def generate_block(in_channels, dim, config):
    block = []
    n_conv_layers = random.choice(
        config['generation']['n_conv_layers_in_block']
    )
    activation_type = random.choice(config['generation']['activations'])

    for i in range(n_conv_layers):
        out_channels = in_channels * 2
        conv_layer = generate_conv_layer(in_channels, out_channels, config)
        in_channels = out_channels
        dim = get_conv_out_dim(conv_layer, dim)
        block.append(conv_layer)
        activation_layer = nn.ReLU() if activation_type == 'ReLU' else nn.LeakyReLU()
        block.append(activation_layer)

    pooling_layer = generate_pooling_layer(config)
    block.append(pooling_layer)
    dim = get_pooling_out_dim(pooling_layer, dim)
    return block, out_channels, dim


def generate_linear_block(in_channels, in_dim, config):
    block = []
    n_linear_layers = random.choice(
        config['generation']['n_linear_layers_in_block']
    )
    in_features = (in_channels * in_dim[0] * in_dim[1]).item()

    block.append(nn.Flatten())
    out_features = random.randint(30, 300)
    for i in range(n_linear_layers):
        fc_layer = nn.Linear(
            in_features=in_features,
            out_features=out_features
        )
        in_features = out_features
        out_features //= 2
        block.append(fc_layer)
    return block


class RandomCNN(nn.Module):
    def __init__(self, w_dim, h_dim, config):
        super().__init__()
        self.config = config
        modules = []
        in_channels = 3
        dim = torch.tensor((w_dim, h_dim))
        n_blocks = random.choice(config['generation']['n_blocks'])

        for i in range(n_blocks):
            block, in_channels, dim = generate_block(
                in_channels,
                dim,
                self.config
            )
            modules.extend(block)
        modules.extend(generate_linear_block(in_channels, dim, self.config))

        self.model = nn.Sequential(*modules)

    def __hash__(self) -> int:
        return hash(str(self.model))

    def __eq__(self, other):
        return str(self.model) == str(other.model)

    def __str__(self):
        return str(self.model)

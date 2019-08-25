# Copyright 2018 Vincent Deuschle. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..layer import (Activation,
                     Loss,
                     Flatten,
                     Pooling,
                     Convolution,
                     FullyConnected,
                     LSTM)


def bp2torch_activation(activation: str) -> Activation:
    if activation == 'relu':
        return F.relu
    elif activation == 'sigmoid':
        return F.sigmoid
    elif activation == 'tanh':
        return F.tanh
    elif activation == 'linear':
        return lambda x: x
    else:
        raise RuntimeError(f"unexpected activation type {activation} in bp2torch_activation")


def bp2torch_loss(loss: str) -> Loss:
    if loss == 'cross_entropy':
        return torch.nn.CrossEntropyLoss()
    elif loss == 'mean_squared_error':
        return torch.nn.MSELoss()
    else:
        raise RuntimeError(f"unexpected loss type {loss} in bp2torch_loss")


def get_flatten_layer(shape: Tuple[int, ...]):
    return Flatten(shape)


def get_pooling_layer(shape: Tuple[int, ...], kernel_size: Tuple[int, int]):
    return Pooling(kernel_size, shape)


def get_fully_connected_layer(shape: Tuple[int, ...],
                              input_shape: Tuple[int, ...],
                              size: int,
                              activation: str,
                              ctx: str,
                              params: Optional[List[np.ndarray]] = None) -> FullyConnected:
    if params is None:
        he_init_variance = np.sqrt(2 / input_shape[-1])
        weights = torch.tensor(np.random.randn(input_shape[-1], size)
                               * he_init_variance,
                               device=ctx,
                               dtype=torch.float,
                               requires_grad=True)
        bias = torch.tensor(np.zeros(size),
                            device=ctx,
                            dtype=torch.float,
                            requires_grad=True)
    else:
        weights = torch.tensor(params[0],
                               device=ctx,
                               dtype=torch.float,
                               requires_grad=True)
        bias = torch.tensor(params[1],
                            device=ctx,
                            dtype=torch.float,
                            requires_grad=True)
    activation = bp2torch_activation(activation)
    return FullyConnected(weights, bias, activation, (input_shape[-1], shape[-1]))


def get_convolution_layer(shape: Tuple[int, ...],
                          input_shape: Tuple[int, ...],
                          channel: int,
                          kernel_size: Tuple[int, int],
                          activation: str,
                          ctx: str,
                          params: Optional[List[np.ndarray]] = None) -> Convolution:
    if params is None:
        he_init_variance = np.sqrt(2 / (kernel_size[0] * kernel_size[1] * input_shape[-1]))
        weights = torch.tensor(np.random.randn(channel, input_shape[-1], *kernel_size)
                               * he_init_variance,
                               device=ctx,
                               dtype=torch.float,
                               requires_grad=True)
        bias = torch.tensor(np.zeros((shape[-1], 1, 1)),
                            device=ctx,
                            dtype=torch.float,
                            requires_grad=True)
    else:
        weights = torch.tensor(params[0],
                               device=ctx,
                               dtype=torch.float,
                               requires_grad=True)
        bias = torch.tensor(params[1],
                            device=ctx,
                            dtype=torch.float,
                            requires_grad=True)
    activation = bp2torch_activation(activation)
    return Convolution(weights, bias, activation, (shape[-1], input_shape[-1], *kernel_size))


def get_lstm_layer(shape: Tuple[int, ...],
                   input_shape: Tuple[int, ...],
                   size: int,
                   return_full_seq: bool,
                   ctx: str,
                   params: Optional[List[np.ndarray]] = None) -> LSTM:
    cell = torch.nn.LSTMCell(input_shape[-1], size)
    if params is None:
        he_init_variance = np.sqrt(2 / input_shape[-1])
        in_shape = 4*size
        cell.weight_ih.data = torch.tensor(np.random.randn(in_shape, input_shape[-1])
                                           * he_init_variance,
                                           device=ctx,
                                           dtype=torch.float,
                                           requires_grad=True)
        cell.bias_ih.data = torch.tensor(np.zeros(in_shape),
                                         device=ctx,
                                         dtype=torch.float,
                                         requires_grad=True)
        cell.weight_hh.data = torch.tensor(np.random.randn(in_shape, size)
                                           * he_init_variance,
                                           device=ctx,
                                           dtype=torch.float,
                                           requires_grad=True)
        cell.bias_hh.data = torch.tensor(np.zeros(in_shape),
                                         device=ctx,
                                         dtype=torch.float,
                                         requires_grad=True)
    else:
        for param, weights in zip(params[::-1], cell.parameters()):
            weights.data = torch.tensor(param,
                                        device=ctx,
                                        dtype=torch.float,
                                        requires_grad=True)
    if ctx == 'cuda':
        cell = torch.nn.LSTMCell(input_shape[-1], size).cuda()
    return LSTM(cell, return_full_seq, (input_shape[-1], *shape))

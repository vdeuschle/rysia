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

from typing import Tuple

import mxnet as mx

from ..layer import (Activation,
                     Loss,
                     Flatten,
                     Pooling,
                     Convolution,
                     FullyConnected,
                     LSTM)


def bp2mxnet_activation(activation: str) -> Activation:
    if activation == 'relu':
        return mx.sym.relu
    elif activation == 'sigmoid':
        return mx.sym.sigmoid
    elif activation == 'tanh':
        return mx.sym.tanh
    elif activation == 'linear':
        return lambda x: x
    else:
        raise RuntimeError(f'Unexpected activation type {activation} in bp2mxnet_activation')


def bp2mxnet_loss(loss: str) -> Loss:
    if loss == 'cross_entropy':
        return mx.sym.SoftmaxOutput
    elif loss == 'mean_squared_error':
        return mx.sym.LinearRegressionOutput
    else:
        raise RuntimeError(f'Unexpected loss type {loss} in bp2mxnet_loss')


def get_flatten_layer(shape: Tuple[int, ...]) -> Flatten:
    return Flatten(shape)


def get_pooling_layer(shape: Tuple[int, ...], kernel_size: Tuple[int, int]) -> Pooling:
    return Pooling(kernel_size, shape)


def get_fully_connected_layer(shape: Tuple[int, ...],
                              input_shape: Tuple[int, ...],
                              size: int, activation: str,
                              layer_index: int) -> FullyConnected:
    weights = mx.sym.Variable(f'fc_weight_params_{layer_index}', shape=(input_shape[-1], size))
    bias = mx.sym.Variable(f'fc_bias_params_{layer_index}', shape=(size,))
    activation = bp2mxnet_activation(activation)
    return FullyConnected(weights, bias, activation, (input_shape[-1], shape[-1]))


def get_convolution_layer(shape: Tuple[int, ...],
                          input_shape: Tuple[int, ...],
                          channel: int,
                          kernel_size: Tuple[int, int],
                          activation: str,
                          layer_index: int) -> Convolution:
    weights = mx.sym.Variable(f'conv_weight_params_{layer_index}',
                              shape=(channel, input_shape[-1], *kernel_size))
    bias = mx.sym.Variable(f'conv_bias_params_{layer_index}',
                           shape=(channel, 1, 1))
    activation = bp2mxnet_activation(activation)
    return Convolution(weights, bias, activation, (shape[-1], input_shape[-1], *kernel_size))


def get_lstm_layer(shape: Tuple[int, ...],
                   input_shape: Tuple[int, ...],
                   size: int,
                   return_full_seq: bool,
                   layer_index: int) -> LSTM:
    cell = mx.rnn.LSTMCell(size, prefix=str(layer_index))
    return LSTM(cell, return_full_seq, (input_shape[-1], *shape))

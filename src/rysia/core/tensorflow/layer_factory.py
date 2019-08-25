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
import tensorflow as tf

from ..layer import (Activation,
                     Loss,
                     Flatten,
                     Pooling,
                     Convolution,
                     FullyConnected,
                     LSTM)


def bp2tf_activation(activation: str) -> Activation:
    if activation == 'relu':
        return tf.nn.relu
    elif activation == 'sigmoid':
        return tf.nn.sigmoid
    elif activation == 'tanh':
        return tf.nn.tanh
    elif activation == 'linear':
        return lambda x: x
    else:
        raise RuntimeError(f'Unexpected activation type {activation} in bp2tf_activation')


def bp2tf_loss(loss: str) -> Loss:
    if loss == 'cross_entropy':
        return tf.losses.sparse_softmax_cross_entropy
    elif loss == 'mean_squared_error':
        return tf.losses.mean_squared_error
    else:
        raise RuntimeError(f'Unexpected loss type {loss} in bp2tf_loss')


def get_flatten_layer(shape: Tuple[int, ...]):
    return Flatten(shape)


def get_pooling_layer(shape: Tuple[int, ...], kernel_size: Tuple[int, int]):
    return Pooling(kernel_size, shape)


def get_fully_connected_layer(shape: Tuple[int, ...],
                              input_shape: Tuple[int, ...],
                              size: int, activation: str,
                              ctx: str,
                              params: Optional[List[np.ndarray]] = None) -> FullyConnected:
    if params is None:
        he_init_variance = np.sqrt(2/input_shape[-1])
        with tf.device(ctx):
            weights = tf.Variable(np.random.randn(input_shape[-1], size)
                                  * he_init_variance,
                                  dtype=tf.float32)
            bias = tf.Variable(np.zeros(size), dtype=tf.float32)
    else:
        with tf.device(ctx):
            weights = tf.Variable(params[0], dtype=tf.float32)
            bias = tf.Variable(params[1], dtype=tf.float32)
    activation = bp2tf_activation(activation)
    return FullyConnected(weights, bias, activation, (input_shape[-1], shape[-1]))


def get_convolution_layer(shape: Tuple[int, ...],
                          input_shape: Tuple[int, ...], #
                          channel: int,
                          kernel_size: Tuple[int, int],
                          activation: str,
                          ctx: str,
                          params: Optional[List[np.ndarray]] = None) -> Convolution:
    if params is None:
        he_init_variance = np.sqrt(2/(kernel_size[0] * kernel_size[1] * input_shape[-1]))
        with tf.device(ctx):
            weights = tf.Variable(np.random.randn(*kernel_size, input_shape[-1], channel)
                                  * he_init_variance,
                                  dtype=tf.float32)
            bias = tf.Variable(np.zeros(shape[-1]), dtype=tf.float32)
    else:
        with tf.device(ctx):
            weights = tf.Variable(params[0], dtype=tf.float32)
            bias = tf.Variable(params[1], dtype=tf.float32)
    activation = bp2tf_activation(activation)
    return Convolution(weights, bias, activation, (shape[-1], input_shape[-1], *kernel_size))


def get_lstm_layer(shape: Tuple[int, ...],
                   input_shape: Tuple[int, ...],
                   size: int,
                   return_full_seq: bool,
                   ctx: str,
                   params: Optional[List[np.ndarray]] = None) -> LSTM:
    if params is None:
        he_init_variance = np.sqrt(2/input_shape[-1])
        weights = (np.random.randn(input_shape[-1]+size, 4*size)
                   * he_init_variance)
        bias = np.zeros(4*size)
        params = [weights, bias]
    with tf.device(ctx):
        cell = (tf.contrib.rnn.LSTMBlockCell(size)
                if ctx == '/cpu:0' else tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(size))
        cell.build(tf.TensorShape((None, input_shape[-1])))
        cell.set_weights(params)
    return LSTM(cell, return_full_seq, (input_shape[-1], *shape))

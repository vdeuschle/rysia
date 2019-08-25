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

from typing import Optional, Tuple, List, Callable

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple

from .. import blueprint as bp
from ..layer import (Model,
                     Sng,
                     Cons,
                     Layer,
                     FullyConnected,
                     Convolution,
                     LSTM,
                     Flatten,
                     Pooling)
from .layer_factory import (get_flatten_layer,
                            get_pooling_layer,
                            get_fully_connected_layer,
                            get_convolution_layer,
                            get_lstm_layer)


def bp2tf_layer(layer: bp.Layer,
                ctx: Optional[str] = None,
                input_shape: Optional[Tuple[int, ...]] = None,
                params: Optional[np.ndarray] = None) -> Layer:
    if isinstance(layer, bp.Flatten):
        shape = layer.shape
        return get_flatten_layer(shape)
    elif isinstance(layer, bp.Pooling):
        shape = layer.shape
        kernel_size = layer.kernel_size
        return get_pooling_layer(shape, kernel_size)
    elif isinstance(layer, bp.FullyConnected):
        shape = layer.shape
        size = layer.size
        activation = layer.activation
        return get_fully_connected_layer(shape, input_shape, size, activation, ctx, params)
    elif isinstance(layer, bp.Convolution):
        shape = layer.shape
        channel = layer.channel
        kernel_size = layer.kernel_size
        activation = layer.activation
        return get_convolution_layer(shape, input_shape, channel, kernel_size, activation, ctx, params)
    elif isinstance(layer, bp.LSTM):
        shape = layer.shape
        size = layer.size
        return_full_seq = layer.return_full_seq
        return get_lstm_layer(shape, input_shape, size, return_full_seq, ctx, params)
    else:
        raise RuntimeError(f'Unexpected layer type {type(layer).__name__} in bp2tf_layer')


def bp2tf_model(model: bp.Model,
                ctx: str,
                params: Optional[List[np.ndarray]] = None) -> Model:
    if isinstance(model, bp.Sng):
        raise RuntimeError('Model must start with input layer and have at least two layer')
    elif isinstance(model, bp.Cons):
        if params is None or isinstance(model.head, (bp.Flatten, bp.Pooling, bp.InputLayer)):
            head = bp2tf_layer(model.head, ctx, model.tail.shape())
        else:
            head = bp2tf_layer(model.head, ctx, model.tail.shape(), params[-2:])
            params = params[:-2]
        if isinstance(model.tail, bp.Sng):
            return Sng(head)
        elif isinstance(model.tail, bp.Cons):
            tail = bp2tf_model(model.tail, ctx, params)
            return Cons(head, tail)
        else:
            raise RuntimeError(f'Unexpected model type {type(model).__name__} in bp2tf_model')
    else:
        raise RuntimeError(f'Unexpected model type {type(model).__name__} in bp2tf_model')


def tf_model2list(model: bp.Model) -> List[tf.Variable]:
    layer = model.head
    if isinstance(layer, (FullyConnected, Convolution)):
        weights = layer.weights
        bias = layer.bias
        params = [bias, weights]
    elif isinstance(layer, LSTM):
        weigths = layer.cell.variables[0]
        bias = layer.cell.variables[1]
        params = [bias, weigths]
    else:
        params = []
    if isinstance(model, Sng):
        return params
    elif isinstance(model, Cons):
        return params + tf_model2list(model.tail)
    else:
        raise RuntimeError(f'Unexpected model type {type(model).__name__} in tf_model2list')


def get_eval_layer(layer: Layer) -> Callable[[tf.Tensor,
                                              Optional[int],
                                              Optional[List[tf.Tensor]]], List[tf.Tensor]]:
    if isinstance(layer, Flatten):
        def eval_flatten(input: tf.Tensor) -> List[tf.Tensor]:
            output = tf.reshape(input, (-1, *layer.shape))
            return [output]
        return eval_flatten
    elif isinstance(layer, FullyConnected):
        def eval_fully_connected(input: tf.Tensor) -> List[tf.Tensor]:
            output = tf.matmul(input, layer.weights)
            output  += layer.bias
            output = layer.activation(output)
            return [output]
        return eval_fully_connected
    elif isinstance(layer, Convolution):
        def eval_convolution(input: tf.Tensor) -> List[tf.Tensor]:
            output = tf.nn.conv2d(input,
                                  layer.weights,
                                  strides=[1, 1, 1, 1],
                                  padding='SAME')
            output += layer.bias
            output = layer.activation(output)
            return [output]
        return eval_convolution
    elif isinstance(layer, Pooling):
        def eval_pooling(input: tf.Tensor) -> List[tf.Tensor]:
            output = tf.nn.max_pool(input,
                                    ksize=[1, *layer.kernel_size, 1],
                                    strides=[1, *layer.kernel_size, 1],
                                    padding='SAME')
            return [output]
        return eval_pooling
    elif isinstance(layer, LSTM):
        def eval_lstm(input: tf.Tensor,
                      seq_length: int,
                      state: List[tf.Tensor]) -> List[tf.Tensor]:
            input = tf.reshape(input, [-1, seq_length, layer.shape[0]])
            output, current_state = tf.nn.dynamic_rnn(layer.cell,
                                                      input,
                                                      initial_state=LSTMStateTuple(*state),
                                                      dtype=tf.float32)
            if layer.return_full_seq:
                output = tf.reshape(output, [-1, layer.shape[-1]])
                return [output, *current_state]
            else:
                return [current_state[1], *current_state]
        return eval_lstm
    else:
        raise RuntimeError(f'Unexpected layer type {type(layer).__name__} in eval_layer')


def get_eval_model(model: Model) -> Callable[[tf.Tensor], List[tf.Tensor]]:
    eval_head = get_eval_layer(model.head)
    if isinstance(model, Sng):
        if isinstance(model.head, LSTM):
            return lambda input, seq_length, state: eval_head(input, seq_length, state)
        else:
            return lambda input, seq_length, state: eval_head(input)
    elif isinstance(model, Cons):
        eval_tail = get_eval_model(model.tail)
        if isinstance(model.head, LSTM):
            def eval_model(input: tf.Tensor,
                           seq_length: Optional[int] = None,
                           state: Optional[List[tf.Tensor]] = None) -> List[tf.Tensor]:
                output_tail = eval_tail(input, seq_length, state[:-2])
                output_head = eval_head(output_tail[0], seq_length, state[-2:])
                return output_head + output_tail[1:]
        else:
            def eval_model(input: tf.Tensor,
                           seq_length: Optional[int] = None,
                           state: Optional[List[tf.Tensor]] = None) -> List[tf.Tensor]:
                output_tail = eval_tail(input, seq_length, state)
                output_head = eval_head(output_tail[0])
                return output_head + output_tail[1:]
        return eval_model
    else:
        raise RuntimeError(f'Unexpected model type {type(model).__name__} in get_eval_model')


def update_layer(layer: Layer,
                 gradients: List[tf.Tensor],
                 learning_rate: float) -> List[tf.Variable]:
    if isinstance(layer, (FullyConnected, Convolution)):
        weight_update = layer.weights.assign(layer.weights
                                             - learning_rate
                                             * gradients[0])
        bias_update = layer.bias.assign(layer.bias
                                        - learning_rate
                                        * gradients[1])
        return [bias_update, weight_update]
    elif isinstance(layer, LSTM):
        weight_update = layer.cell.variables[0].assign(layer.cell.variables[0]
                                                       - learning_rate
                                                       * gradients[0])
        bias_update = layer.cell.variables[1].assign(layer.cell.variables[1]
                                                     - learning_rate
                                                     * gradients[1])
        return [bias_update, weight_update]
    else:
        raise RuntimeError(f'Unexpected layer type {type(layer).__name__} in update_layer')


def update_model(model: Model,
                 gradients: List[tf.Tensor],
                 learning_rate: float) -> List[tf.Variable]:
    if isinstance(model.head, (FullyConnected, Convolution, LSTM)):
        head = update_layer(model.head, gradients[:2], learning_rate)
        gradients = gradients[2:]
    else:
        head = []
    if isinstance(model, Sng):
        return head
    elif isinstance(model, Cons):
        tail = update_model(model.tail, gradients, learning_rate)
        return head + tail
    else:
        raise RuntimeError(f'Unexpected model type {type(model).__name__} in update_model')

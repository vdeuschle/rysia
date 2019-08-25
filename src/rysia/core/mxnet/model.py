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

from typing import List, Optional, Tuple, Callable

import mxnet as mx

from ...utils.misc import get_padding
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


def bp2mxnet_layer(layer: bp.Layer,
                   layer_index: Optional[int] = None,
                   input_shape: Optional[Tuple[int]] = None) -> Layer:
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
        return get_fully_connected_layer(shape,
                                         input_shape,
                                         size,
                                         activation,
                                         layer_index)
    elif isinstance(layer, bp.Convolution):
        shape = layer.shape
        channel = layer.channel
        kernel_size = layer.kernel_size
        activation = layer.activation
        return get_convolution_layer(shape,
                                     input_shape,
                                     channel,
                                     kernel_size,
                                     activation,
                                     layer_index)
    elif isinstance(layer, bp.LSTM):
        shape = layer.shape
        size = layer.size
        return_full_seq = layer.return_full_seq
        return get_lstm_layer(shape,
                              input_shape,
                              size,
                              return_full_seq,
                              layer_index)
    else:
        raise RuntimeError(f'Unexpected layer type {type(layer).__name__} in bp2mxnet_layer')


def bp2mxnet_model(model: bp.Model, layer_index: int = 0) -> Model:
    if isinstance(model, bp.Sng):
        raise RuntimeError('Model must start with input layer and have at least two layer')
    elif isinstance(model, bp.Cons):
        head = bp2mxnet_layer(model.head, layer_index, model.tail.shape())
        if isinstance(model.head, (bp.FullyConnected, bp.Convolution, bp.LSTM)):
            layer_index += 1
        if isinstance(model.tail, bp.Sng):
            return Sng(head)
        elif isinstance(model.tail, bp.Cons):
            tail = bp2mxnet_model(model.tail, layer_index)
            return Cons(head, tail)
        else:
            raise RuntimeError(f'Unexpected model type {type(model).__name__} in bp2mxnet_model')
    else:
        raise RuntimeError(f'Unexpected model type {type(model).__name__} in bp2mxnet_model')


def mxnet_model2list(model: bp.Model) -> List[mx.sym.Variable]:
    layer = model.head
    if isinstance(layer, (FullyConnected, Convolution)):
        weights = layer.weights
        bias = layer.bias
        params = [bias, weights]
    elif isinstance(layer, LSTM):
        cell = layer.cell
        params = [cell.params.get('i2h_weight'), cell.params.get('i2h_bias'),
                  cell.params.get('h2h_weight'), cell.params.get('h2h_bias')]
    else:
        params = []
    if isinstance(model, Sng):
        return params
    elif isinstance(model, Cons):
        return params + mxnet_model2list(model.tail)
    else:
        raise RuntimeError(f'Unexpected model type {type(model).__name__} in mxnet_model2list')


def get_eval_layer(layer: Layer) -> Callable[[mx.sym.Variable,
                                              Optional[int],
                                              Optional[List[mx.sym.Variable]]],
                                             List[mx.sym.Variable]]:
    if isinstance(layer, Flatten):
        def eval_flatten(input: mx.sym.Variable) -> List[mx.sym.Variable]:
            output = mx.sym.reshape(input, (-1, *layer.shape))
            return [output]
        return eval_flatten
    elif isinstance(layer, FullyConnected):
        def eval_fully_connected(input: mx.sym.Variable) -> List[mx.sym.Variable]:
            output = mx.sym.linalg_gemm2(input, layer.weights)
            output = mx.sym.broadcast_add(output, layer.bias)
            output = layer.activation(output)
            return [output]
        return eval_fully_connected
    elif isinstance(layer, Convolution):
        num_filter = layer.shape[0]
        kernel_size = layer.shape[2:]
        padding_h = int(kernel_size[0] / 2)
        padding_w = int(kernel_size[1] / 2)
        def eval_convolution(input: mx.sym.Variable) -> List[mx.sym.Variable]:
            output = mx.sym.Convolution(input,
                                        layer.weights,
                                        kernel=kernel_size,
                                        stride=(1, 1),
                                        pad=(padding_h, padding_w),
                                        num_filter=num_filter,
                                        no_bias=True)
            output = mx.sym.broadcast_add(output, layer.bias)
            output = layer.activation(output)
            return [output]
        return eval_convolution
    elif isinstance(layer, Pooling):
        kernel_size = layer.kernel_size
        input_shape = layer.shape[:2]
        padding_h = int(get_padding(kernel_size[0], input_shape[0]))
        padding_w = int(get_padding(kernel_size[1], input_shape[1]))
        def eval_pooling(input: mx.sym.Variable) -> List[mx.sym.Variable]:
            output = mx.sym.Pooling(input,
                                    kernel=kernel_size,
                                    stride=kernel_size,
                                    pad=(padding_h, padding_w))
            return [output]
        return eval_pooling
    elif isinstance(layer, LSTM):
        def eval_lstm(input: mx.sym.Variable,
                      seq_length: int,
                      state: List[mx.sym.Variable]) -> List[mx.sym.Variable]:
            input = mx.sym.reshape(input, (-1, seq_length, layer.shape[0]))
            output, current_state = layer.cell.unroll(seq_length,
                                                      input, state,
                                                      layout='NTC',
                                                      merge_outputs=True)
            if layer.return_full_seq:
                output = mx.sym.reshape(output, [-1, layer.shape[-1]])
                return [output,
                        mx.sym.BlockGrad(current_state[0]),
                        mx.sym.BlockGrad(current_state[1])]
            else:
                return [current_state[0],
                        mx.sym.BlockGrad(current_state[0]),
                        mx.sym.BlockGrad(current_state[1])]
        return eval_lstm
    else:
        raise RuntimeError(f'Unexpected layer type {type(layer).__name__} in eval_layer')


def get_eval_model(model: Model) -> Callable[[mx.sym.Variable], List[mx.sym.Variable]]:
    eval_head = get_eval_layer(model.head)
    if isinstance(model, Sng):
        if isinstance(model.head, LSTM):
            return lambda input, seq_length, state: eval_head(input, seq_length, state)
        else:
            return lambda input, seq_length, state: eval_head(input)
    elif isinstance(model, Cons):
        eval_tail = get_eval_model(model.tail)
        if isinstance(model.head, LSTM):
            def eval_model(input: mx.sym.Variable,
                           seq_length: Optional[int] = None,
                           state: Optional[List[mx.sym.Variable]] = None) -> List[mx.sym.Variable]:
                output_tail = eval_tail(input, seq_length, state[:-2])
                output_head = eval_head(output_tail[0], seq_length, state[-2:])
                return output_head + output_tail[1:]
        else:
            def eval_model(input: mx.sym.Variable,
                           seq_length: Optional[int] = None,
                           state: Optional[List[mx.sym.Variable]] = None) -> List[mx.sym.Variable]:
                output_tail = eval_tail(input, seq_length, state)
                output_head = eval_head(output_tail[0])
                return output_head + output_tail[1:]
        return eval_model
    else:
        raise RuntimeError(f'Unexpected model type {type(model).__name__} in get_eval_model')


def update_layer(layer: Layer,
                 learning_rate: float,
                 norm: int) -> List[mx.sym.Variable]:
    if isinstance(layer, (FullyConnected, Convolution)):
        weights_update = (layer.weights
                          - learning_rate
                          * mx.sym.Variable(f'{layer.weights.name}_grad')
                          / norm)
        bias_update = (layer.bias
                       - learning_rate
                       * mx.sym.Variable(f'{layer.bias.name}_grad')
                       / norm)
        return [weights_update, bias_update]
    elif isinstance(layer, LSTM):
        i2h_weight_update = (layer.cell.params.get('i2h_weight')
                             - learning_rate
                             * mx.sym.Variable(layer.cell.params.get('i2h_weight').name+'_grad')
                             / norm)
        i2h_bias_update = (layer.cell.params.get('i2h_bias')
                           - learning_rate
                           * mx.sym.Variable(layer.cell.params.get('i2h_bias').name+'_grad')
                           / norm)
        h2h_weight_update = (layer.cell.params.get('h2h_weight')
                             - learning_rate
                             * mx.sym.Variable(layer.cell.params.get('h2h_weight').name+'_grad')
                             / norm)
        h2h_bias_update = (layer.cell.params.get('h2h_bias')
                           - learning_rate
                           * mx.sym.Variable(layer.cell.params.get('i2h_bias').name+'_grad')
                           / norm)
        return [i2h_weight_update,
                i2h_bias_update,
                h2h_weight_update,
                h2h_bias_update]
    else:
        raise RuntimeError(f'Layer type {type(layer).__name__} has no gradients')


def update_model(model: Model,
                 learning_rate: float,
                 norm: int) -> List[mx.sym.Variable]:
    if isinstance(model.head, (FullyConnected, Convolution, LSTM)):
        head = update_layer(model.head, learning_rate, norm)
    else:
        head = []
    if isinstance(model, Sng):
        return head
    elif isinstance(model, Cons):
        tail = update_model(model.tail, learning_rate, norm)
        return tail + head
    else:
        raise RuntimeError(f'Unexpected model type {type(model).__name__} in update_model')

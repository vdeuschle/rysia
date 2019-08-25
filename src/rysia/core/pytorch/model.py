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
import torch
import torch.nn.functional as F

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


def bp2torch_layer(layer: bp.Layer,
                   ctx: Optional[str] = None,
                   input_shape: Optional[Tuple[int]] = None,
                   params: Optional[List[np.ndarray]] = None) -> Layer:
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
        raise RuntimeError(f'Unexpected layer type {type(layer).__name__} in bp2torch_layer')


def bp2torch_model(model: bp.Model,
                   ctx: str,
                   params: Optional[List[np.ndarray]] = None) -> Model:
    if isinstance(model, bp.Sng):
        raise RuntimeError('Model must start with input layer and have at least two layer')
    elif isinstance(model, bp.Cons):
        if params is None or isinstance(model.head, (bp.Flatten, bp.Pooling, bp.InputLayer)):
            head = bp2torch_layer(model.head, ctx, model.tail.shape())
        elif isinstance(model.head, (bp.FullyConnected, bp. Convolution)):
            head = bp2torch_layer(model.head, ctx, model.tail.shape(), params[-2:])
            params = params[:-2]
        elif isinstance(model.head, bp.LSTM):
            head = bp2torch_layer(model.head, ctx, model.tail.shape(), params[-4:])
            params = params[:-4]
        else:
            head = bp2torch_layer(model.head, ctx, model.tail.shape())
        if isinstance(model.tail, bp.Sng):
            return Sng(head)
        elif isinstance(model.tail, bp.Cons):
            tail = bp2torch_model(model.tail, ctx, params)
            return Cons(head, tail)
        else:
            raise RuntimeError(f'Unexpected model type {type(model).__name__} in bp2torch_model')
    else:
        raise RuntimeError(f'Unexpected model type {type(model).__name__} in bp2torch_model')


def torch_model2list(model: Model) -> List[torch.Tensor]:
    layer = model.head
    if isinstance(layer, (FullyConnected, Convolution)):
        weights = layer.weights
        bias = layer.bias
        params = [bias, weights]
    elif isinstance(layer, LSTM):
        params = list(layer.cell.parameters())
    else:
        params = []
    if isinstance(model, Sng):
        return params
    elif isinstance(model, Cons):
        return params + torch_model2list(model.tail)
    else:
        raise RuntimeError(f'Unexpected core type {type(model).__name__} in torch_model2list')


def get_eval_layer(layer: Layer) -> Callable[[torch.Tensor, Optional[int],
                                              Optional[List[torch.Tensor]]],
                                             List[torch.Tensor]]:
    if isinstance(layer, Flatten):
        def eval_flatten(input: torch.Tensor) -> List[torch.Tensor]:
            output = input.view(-1, *layer.shape)
            return [output]
        return eval_flatten
    elif isinstance(layer, FullyConnected):
        def eval_fully_connected(input: torch.Tensor) -> List[torch.Tensor]:
            output = input @ layer.weights
            output += layer.bias
            output = layer.activation(output)
            return [output]
        return eval_fully_connected
    elif isinstance(layer, Convolution):
        kernel_size = layer.shape[2:]
        padding_h = int(kernel_size[0] / 2)
        padding_w = int(kernel_size[1] / 2)
        def eval_convolution(input: torch.Tensor) -> List[torch.Tensor]:
            output = F.conv2d(input,
                              layer.weights,
                              stride=(1,1),
                              padding=(padding_h, padding_w))
            output += layer.bias
            output = layer.activation(output)
            return [output]
        return eval_convolution
    elif isinstance(layer, Pooling):
        kernel_size = layer.kernel_size
        input_shape = layer.shape[:2]
        padding_h = int(get_padding(kernel_size[0], input_shape[0]))
        padding_w = int(get_padding(kernel_size[1], input_shape[1]))
        def eval_pooling(input: torch.Tensor) -> List[torch.Tensor]:
            output = F.max_pool2d(input,
                                  kernel_size=kernel_size,
                                  stride=kernel_size,
                                  padding=(padding_h, padding_w))
            return [output]
        return eval_pooling
    elif isinstance(layer, LSTM):
        def eval_lstm(input: torch.Tensor,
                      seq_length: int,
                      state: List[torch.Tensor]) -> List[torch.Tensor]:
            input = input.view(-1, seq_length, layer.shape[0])
            if layer.return_full_seq:
                output = torch.empty((input.shape[0], seq_length, layer.shape[-1]),
                                     dtype=torch.float,
                                     device=input.device)
                for i in range(seq_length):
                    state = layer.cell(input[:,i,:], state)
                    output[:,i,:] = state[0]
                output = output.view(-1, layer.shape[-1])
                return [output, *state]
            else:
                for i in range(seq_length):
                    state = layer.cell(input[:,i,:], state)
                return [state[0], *state]
        return eval_lstm
    else:
        raise RuntimeError(f'Unexpected layer type {type(layer).__name__} in get_eval_layer')


def get_eval_model(model: Model) -> Callable[[torch.Tensor, Optional[int],
                                              Optional[List[torch.Tensor]]],
                                             List[torch.Tensor]]:
    eval_head = get_eval_layer(model.head)
    if isinstance(model, Sng):
        if type(model.head) is LSTM:
            return lambda input, seq_length, state: eval_head(input, seq_length, state)
        else:
            return lambda input, seq_length, state: eval_head(input)
    elif isinstance(model, Cons):
        eval_tail = get_eval_model(model.tail)
        if isinstance(model.head, LSTM):
            def eval_model(input: torch.Tensor, seq_length: Optional[int] = None,
                           state: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
                output_tail = eval_tail(input, seq_length, state[:-2])
                output_head = eval_head(output_tail[0], seq_length, state[-2:])
                return output_head + output_tail[1:]
        else:
            def eval_model(input: torch.Tensor, seq_length: Optional[int] = None,
                           state: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
                output_tail = eval_tail(input, seq_length, state)
                output_head = eval_head(output_tail[0])
                return output_head + output_tail[1:]
        return eval_model
    else:
        raise RuntimeError(f'Unexpected core type {type(model).__name__} in get_eval_model')


def update_layer(layer: Layer, learning_rate: float) -> None:
    if isinstance(layer, (FullyConnected, Convolution)):
        layer.weights.data -= learning_rate * layer.weights.grad.data
        layer.bias.data -= learning_rate * layer.bias.grad.data
        layer.weights.grad.zero_()
        layer.bias.grad.zero_()
    elif isinstance(layer, LSTM):
        for param in layer.cell.parameters():
            param.data -= learning_rate * param.grad.data
            param.grad.zero_()
    else:
        raise RuntimeError(f'Unexpected layer type {type(layer).__name__} in update_layer')


def update_model(model: Model, learning_rate: float) -> None:
    if isinstance(model.head, (FullyConnected, Convolution, LSTM)):
        update_layer(model.head, learning_rate)
    if isinstance(model, Cons):
        update_model(model.tail, learning_rate)

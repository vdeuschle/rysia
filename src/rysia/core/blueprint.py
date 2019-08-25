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

from typing import NamedTuple, Optional, Tuple, List

from ..utils.misc import prod, get_padding


class Layer(NamedTuple):
    pass


class InputLayer(Layer, NamedTuple):
    shape: Optional[Tuple[int, ...]] = None


class Flatten(Layer, NamedTuple):
    flatten_sequences: bool = False
    shape: Optional[Tuple[int, ...]] = None


class FullyConnected(Layer, NamedTuple):
    size: int
    activation: str
    shape: Optional[Tuple[int, ...]] = None


class Convolution(Layer, NamedTuple):
    channel: int
    kernel_size: Tuple[int, int]
    activation: str
    shape: Optional[Tuple[int, ...]] = None

class Pooling(Layer, NamedTuple):
    kernel_size: Tuple[int, int]
    shape: Optional[Tuple[int, ...]] = None


class LSTM(Layer, NamedTuple):
    size: int
    return_full_seq: bool
    shape: Optional[Tuple[int, ...]] = None


class Model(NamedTuple):
    def shape(self):
        pass


class Sng(Model, NamedTuple):
    head: Layer

    def shape(self):
        return self.head.shape


class Cons(Model, NamedTuple):
    head: Layer
    tail: Model

    def shape(self):
        return self.head.shape


def infer_shape_layer(layer: Layer, input_shape: Optional[Tuple[int, ...]] = None) -> Layer:
    if isinstance(layer, InputLayer):
        shape = layer.shape
        return InputLayer(shape)
    elif isinstance(layer, Flatten):
        flatten_sequences = layer.flatten_sequences
        shape = (int(prod(input_shape)),) if not layer.flatten_sequences else (input_shape[-1])
        return Flatten(flatten_sequences, shape)
    elif isinstance(layer, FullyConnected):
        size = layer.size
        activation = layer.activation
        shape = (layer.size,)
        return FullyConnected(size, activation, shape)
    elif isinstance(layer, Convolution):
        kernel_size = layer.kernel_size
        if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
            raise RuntimeError('Only odd kernel sizes are supported for convolution')
        out_channel = layer.channel
        activation = layer.activation
        shape = (*input_shape[:-1], out_channel)
        return Convolution(out_channel, kernel_size, activation, shape)
    elif isinstance(layer, Pooling):
        kernel_size = layer.kernel_size
        padding_h = get_padding(kernel_size[0], input_shape[0])
        padding_w = get_padding(kernel_size[1], input_shape[1])
        if padding_h != int(padding_h) or padding_w != int(padding_w):
            raise RuntimeError('2DPooling must require an even padding number')
        width = input_shape[0] / kernel_size[0]
        width = width if int(width) == width else int(width) + 1
        height = input_shape[1] / kernel_size[1]
        height = height if int(height) == height else int(height) + 1
        shape = (width, height, *input_shape[2:])
        return Pooling(layer.kernel_size, shape)
    elif isinstance(layer, LSTM):
        size = layer.size
        return_full_seq = layer.return_full_seq
        shape = (layer.size,)
        return LSTM(size, return_full_seq, shape)
    else:
        raise RuntimeError(f'Unexpected layer type {type(layer).__name__} in infer_shape_layer')


def infer_shape_model(model: Model) -> Model:
    if isinstance(model, Sng):
        return Sng(infer_shape_layer(model.head, None))
    elif isinstance(model, Cons):
        tail = infer_shape_model(model.tail)
        head = infer_shape_layer(model.head, tail.shape())
        return Cons(head, tail)
    else:
        raise RuntimeError(f'Unexpected model type {type(model).__name__} in infer_shape_model')


def list2model(ls: List[Layer]) -> Model:
    if len(ls) < 1:
        raise RuntimeError('Cannot convert empty list to a model')
    elif len(ls) == 1:
        return Sng(head=ls[-1])
    else:
        return Cons(head=ls[-1], tail=list2model(ls[:-1]))


def model(*ls: List[Layer]) -> Model:
    return infer_shape_model(list2model(ls))
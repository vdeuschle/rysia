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

from typing import (NamedTuple,
                    Tuple,
                    Callable,
                    Optional,
                    TypeVar,
                    Union)

Variable = TypeVar('Variable',
                   'torch.tensor',
                   'tensorflow.Variable',
                   'mxnet.sym.Variable')

LSTMCell = TypeVar('LSTMCell',
                   Union['tensorflow.contrib.rnn.LSTMBlockCell',
                         'tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell'],
                   'torch.nn.LSTMCell',
                   'mxnet.rnn.LSTMCell')

Activation = Callable[[Variable], Variable]

Loss = Callable[[Variable, Variable], Variable]


class Layer(NamedTuple):
    shape: Optional[Tuple[int, ...]]


class Flatten(Layer, NamedTuple):
    shape: Optional[Tuple[int, ...]]


class FullyConnected(Layer, NamedTuple):
    weights: Variable
    bias: Variable
    activation: Activation
    shape: Optional[Tuple[int, ...]]


class Convolution(Layer, NamedTuple):
    weights: Variable
    bias: Variable
    activation: Activation
    shape: Optional[Tuple[int, ...]]


class LSTM(Layer, NamedTuple):
    cell: LSTMCell
    return_full_seq: bool
    shape: Optional[Tuple[int, ...]]


class Pooling(Layer, NamedTuple):
    kernel_size: Tuple[int, int]
    shape: Optional[Tuple[int, ...]]


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

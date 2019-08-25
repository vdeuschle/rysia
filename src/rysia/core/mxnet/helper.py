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

from typing import List

import numpy as np
import mxnet as mx

from ...utils.misc import prod
from ..layer import (Model,
                     Sng,
                     Cons,
                     FullyConnected,
                     Convolution,
                     LSTM,
                     Flatten,
                     Pooling)


def init_params(model: Model, ctx: mx.Context) -> List[mx.nd.NDArray]:
        layer = model.head
        if isinstance(layer, FullyConnected):
            he_init_variance = np.sqrt(2 / layer.shape[0])
            params = [mx.nd.array(np.random.randn(*layer.shape) * he_init_variance,
                                  dtype='float32', ctx=ctx),
                      mx.nd.array(np.zeros(layer.shape[-1]),
                                  dtype='float32', ctx=ctx)]
        elif isinstance(layer, Convolution):
            he_init_variance = np.sqrt(2 / prod(layer.shape[:-1]))
            params = [mx.nd.array(np.random.randn(*layer.shape) * he_init_variance,
                                  dtype='float32', ctx=ctx),
                      mx.nd.array(np.zeros((layer.shape[0], 1, 1)),
                                  dtype='float32', ctx=ctx)]
        elif isinstance(layer, LSTM):
            he_init_variance = np.sqrt(2 / layer.shape[0])
            out_shape = 4 * layer.shape[-1]
            params = [mx.nd.array(np.random.randn(out_shape, layer.shape[0]) * he_init_variance,
                                  dtype='float32', ctx=ctx),
                      mx.nd.array(np.zeros(out_shape),
                                  dtype='float32', ctx=ctx),
                      mx.nd.array(np.random.randn(out_shape, layer.shape[-1]) * he_init_variance,
                                  dtype='float32', ctx=ctx),
                      mx.nd.array(np.zeros(out_shape),
                                  dtype='float32', ctx=ctx)]
        elif isinstance(layer, (Flatten, Pooling)):
            params = []
        else:
            raise RuntimeError(f'Unexpected layer type {type(layer).__name__} in mx_model2list')
        if isinstance(model, Sng):
            return params
        elif isinstance(model, Cons):
            return init_params(model.tail, ctx) + params
        else:
            raise RuntimeError(f'Unexpected model type {type(model).__name__} in mx_model2list')


def load_params(params: List[np.ndarray], ctx: mx.Context) -> List[mx.nd.NDArray]:
    params_ndarray = list(map(lambda x: mx.nd.array(x, dtype='float32', ctx=ctx), params))
    return params_ndarray


def get_optimizer(optimizer: str, learning_rate: float, batch_size: int):
    if optimizer == 'sgd':
        return mx.optimizer.SGD(learning_rate=learning_rate, rescale_grad=1/batch_size)
    elif optimizer == 'adam':
        return mx.optimizer.Adam(learning_rate=learning_rate, rescale_grad=1/batch_size)
    else:
        raise RuntimeError(f'Unknown optimizer {optimizer} in get_train_step')

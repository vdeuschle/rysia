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

import time
from typing import List

import numpy as np
import mxnet as mx

from ..layer import Model
from .helper import init_params, load_params
from .model import get_eval_model


def inference(model: Model,
              ctx: mx.Context,
              params: List[np.ndarray],
              data: np.ndarray,
              batch_size: int) -> List[float]:
    data_size = data.shape[0]
    input_dim = data.shape[1:]

    eval_model = get_eval_model(model)
    prediction = eval_model(mx.sym.Variable('data'))[0]
    prediction = mx.sym.softmax(prediction)
    if params is None:
        model_params = init_params(model, ctx)
    else:
        model_params = load_params(params, ctx)
    predict_args = dict(zip(prediction.list_arguments(),
                            [mx.nd.empty((batch_size, *input_dim),
                                         ctx=ctx,
                                         dtype='float32')]
                            + model_params))
    executor = prediction.bind(ctx, args=predict_args)

    counter = 0
    stop = time.time() + 60
    while time.time() < stop:
        idx = np.random.choice(data_size, batch_size)
        test_batch = data[idx]
        predict_args['data'][:] = mx.nd.array(test_batch,
                                              ctx=ctx,
                                              dtype='float32')
        output = executor.forward(is_train=False)[0].asnumpy()
        counter += 1

    bps = counter / 60
    return [stop - 60, stop, bps]


def recurrent_inference(model: Model,
                        ctx: mx.Context,
                        params: List[np.ndarray],
                        data: np.ndarray,
                        batch_size: int,
                        state_sizes: List[int]) -> List[float]:
    data_size = data.shape[0]
    seq_length = data.shape[1]
    input_dim = data.shape[1:]

    states = [mx.sym.Variable(f'{state}_state_%i' % i)
              for i, _ in enumerate(state_sizes)
              for state in ['hidden', 'cell']]
    eval_model = get_eval_model(model)
    prediction = eval_model(mx.sym.Variable('data'), seq_length, states)[0]
    prediction = mx.sym.softmax(prediction)
    if params is None:
        model_params = init_params(model, ctx)
    else:
        model_params = load_params(params, ctx)
    param_names = list(filter(lambda x: 'state' not in x,
                              prediction.list_arguments()[1:]))
    init_states = {state.name: mx.nd.empty((batch_size, state_sizes[i // 2]),
                                           ctx=ctx,
                                           dtype='float32')
                   for i, state in enumerate(states)}
    args = {**dict(zip(param_names, model_params)), **init_states,
            'data': mx.nd.empty((batch_size, *input_dim),
                                ctx=ctx,
                                dtype='float32')}
    executor = prediction.bind(ctx, args=args)

    counter = 0
    stop = time.time() + 60
    while time.time() < stop:
        idx = np.random.choice(data_size, batch_size)
        test_batch = data[idx]
        args['data'][:] = mx.nd.array(test_batch,
                                      ctx=ctx,
                                      dtype='float32')
        output = executor.forward(is_train=False)[0].asnumpy()
        counter += 1

    bps = counter / 60
    return [stop - 60, stop, bps]

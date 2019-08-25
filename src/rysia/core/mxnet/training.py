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

import logging
from typing import Optional, List

import numpy as np
import mxnet as mx
from mxnet import Context

from ...utils.misc import accuracy
from ..layer import Model
from .model import get_eval_model
from .layer_factory import bp2mxnet_loss
from .helper import init_params, get_optimizer


def stochastic_gradient_descent(model: Model,
                                ctx: Context,
                                train_data: np.ndarray,
                                train_label: np.ndarray,
                                test_data: Optional[np.ndarray],
                                test_label: Optional[np.ndarray],
                                loss_func: str,
                                optimizer: str,
                                num_epochs: int,
                                learning_rate: float,
                                batch_size: Optional[int]) -> List[float]:
    train_label = (train_label.argmax(axis=1)
                   if loss_func == 'cross_entropy'
                   else train_label)
    train_data_size = train_data.shape[0]
    input_dim = train_data.shape[1:]
    if batch_size is None or batch_size > train_data_size:
        batch_size = train_data_size
    loss_function = bp2mxnet_loss(loss_func)
    eval_model = get_eval_model(model)
    prediction = eval_model(mx.sym.Variable('data'))[0]
    loss = loss_function(prediction, mx.sym.Variable('label'))

    param_names = loss.list_arguments()[1:-1]
    model_params = init_params(model, ctx)
    model_optimizer = get_optimizer(optimizer, learning_rate, batch_size)
    train_module = mx.mod.Module(loss, context=ctx, label_names=['label'])
    train_module.bind(data_shapes=[('data', (batch_size, *input_dim))],
                      label_shapes=[('label', (batch_size, ))])
    train_module.set_params(dict(zip(param_names, model_params)), aux_params={})
    train_module.init_optimizer(optimizer=model_optimizer)
    all_accuracies = [['train accuracy']]

    testing = not (test_data is None or test_label is None)
    if testing:
        test_data = mx.nd.array(test_data, ctx=ctx, dtype='float32')
        test_module = mx.mod.Module(prediction)
        test_module.bind(data_shapes=[('data', test_data.shape)])
        all_accuracies[0] += ['test accuracy']

    for i in range(num_epochs):
        current_train_accuracies = []
        perm = np.random.permutation(train_data_size)
        for j in range(0, train_data_size-train_data_size%batch_size, batch_size):
            idx = perm[j:j+batch_size]
            current_batch = train_data[idx]
            current_label = train_label[idx]
            train_module.forward(mx.io.DataBatch(data=[mx.nd.array(current_batch,
                                                                   ctx=ctx,
                                                                   dtype='float32')],
                                                 label=[mx.nd.array(current_label,
                                                                    ctx=ctx,
                                                                    dtype='int64')]),
                                 is_train=True)
            train_prediction = train_module.get_outputs()[0]
            train_module.backward()
            train_module.update()
            current_train_accuracies += [accuracy(train_prediction.asnumpy(), current_label)]
        current_train_accuracies_mean = np.array(current_train_accuracies).mean()
        all_accuracies += [[current_train_accuracies_mean]]

        if testing:
            test_module.set_params(*train_module.get_params())
            test_module.forward(mx.io.DataBatch(data=[test_data]), is_train=False)
            test_prediction = test_module.get_outputs()[0]
            test_accuracy = accuracy(test_prediction.asnumpy(), test_label)
            all_accuracies[-1] += [test_accuracy]
            logging.info(
                f'Train accuracy: {current_train_accuracies_mean}; Test accuracy: {test_accuracy}')
        else:
            logging.info(
                f'Train accuracy: {current_train_accuracies_mean}')

    return all_accuracies


def recurrent_gradient_descent(model: Model,
                               ctx: Context,
                               train_data: np.ndarray,
                               train_label: np.ndarray,
                               test_data:
                               Optional[np.ndarray],
                               test_label: Optional[np.ndarray],
                               train_seq_length: int,
                               test_seq_length: int,
                               state_sizes: List[int],
                               truncated_backprop_length: int,
                               loss_func: str,
                               optimizer: str,
                               num_epochs: int,
                               learning_rate: float,
                               batch_size: Optional[int]) -> List[float]:
    train_label = (train_label.argmax(axis=1)
                   if loss_func == 'cross_entropy'
                   else train_label)
    train_data_size = train_data.shape[0]
    input_dim = train_data.shape[2:]
    if batch_size is None or batch_size > train_data_size:
        batch_size = train_data_size
    return_full_seq = len(train_data.shape) == len(train_label.shape)
    iterate_over_seq = train_seq_length > truncated_backprop_length
    loss_function = bp2mxnet_loss(loss_func)

    state_names = [f'{state}_state_%i' % i
                   for i in range(len(state_sizes))
                   for state in ['hidden', 'cell']]
    states = [mx.sym.Variable(state_names[i])
              for i, _ in enumerate(state_names)]
    eval_model = get_eval_model(model)
    prediction = eval_model(mx.sym.Variable('data'),
                            truncated_backprop_length,
                            states)
    loss = loss_function(prediction[0], mx.sym.Variable('label'))
    train_output = mx.sym.Group([loss, *prediction[1:]])

    param_names = list(filter(lambda x: 'state' not in x, loss.list_arguments()[1:-1]))
    model_params = init_params(model, ctx)
    model_optimizer = get_optimizer(optimizer, learning_rate, batch_size)
    state_shapes = [(state_names[i], (batch_size, state_sizes[i//2]))
                    for i, _ in enumerate(state_names)]
    train_module = mx.mod.Module(train_output,
                                 context=ctx,
                                 data_names=['data', *state_names],
                                 label_names=['label'])
    train_module.bind(data_shapes=[('data', (batch_size, truncated_backprop_length, *input_dim)),
                                   *state_shapes],
                      label_shapes=[('label', (batch_size, ))])
    train_module.set_params(dict(zip(param_names, model_params)), aux_params={})
    train_module.init_optimizer(optimizer=model_optimizer)
    all_accuracies = [['train accuracy']]

    testing = not (test_data is None or test_label is None)
    if testing:
        test_data_size = test_data.shape[0]
        test_data = mx.nd.array(test_data, ctx=ctx, dtype='float32')
        init_test_states = [mx.nd.zeros((test_data_size, state_size), dtype='float32')
                            for state_size in state_sizes for _ in range(2)]
        test_module = mx.mod.Module(prediction[0], data_names=['data', *state_names])
        test_module.bind(data_shapes=[('data', test_data.shape),
                                      *list(map(lambda x: (x[0], (test_data_size, x[1][1])),
                                                state_shapes))])
        test_module.set_params(dict(zip(param_names, model_params)), aux_params={})
        all_accuracies[0] += ['test accuracy']

    for i in range(num_epochs):
        current_train_accuracies = []
        perm = np.random.permutation(train_data_size)
        for j in range(0, train_data_size-train_data_size%batch_size, batch_size):
            idx = perm[j:j+batch_size]
            current_batch = train_data[idx]
            current_label = train_label[idx]
            init_train_states = [mx.nd.zeros((batch_size,state_size), dtype='float32')
                                 for state_size in state_sizes for _ in range(2)]
            for k in range(0, train_seq_length, truncated_backprop_length):
                batch_X = current_batch[:, k:k+truncated_backprop_length, :]
                batch_Y = current_label
                if return_full_seq:
                    batch_Y = batch_Y[:, k:k+truncated_backprop_length, :].reshape(-1)
                train_module.forward(mx.io.DataBatch(data=[mx.nd.array(batch_X,
                                                                       ctx=ctx,
                                                                       dtype='float32')]
                                                           + init_train_states,
                                                     label=[mx.nd.array(batch_Y,
                                                                        ctx=ctx,
                                                                        dtype='int64')]),
                                     is_train=True)
                train_prediction = train_module.get_outputs()
                train_module.backward()
                train_module.update()
                if iterate_over_seq:
                    init_train_states = train_prediction[1:]
                current_train_accuracies += [accuracy(train_prediction[0].asnumpy(), batch_Y)]
        current_train_accuracies_mean = np.array(current_train_accuracies).mean()
        all_accuracies += [[current_train_accuracies_mean]]

        if testing:
            test_module.set_params(*train_module.get_params())
            test_module.forward(mx.io.DataBatch(data=[test_data] + init_test_states),
                                is_train=False)
            test_prediction = test_module.get_outputs()[0]
            test_accuracy = accuracy(test_prediction.asnumpy(), test_label)
            all_accuracies[-1] += [test_accuracy]
            logging.info(
                f'Train accuracy: {current_train_accuracies_mean}; Test accuracy: {test_accuracy}')
        else:
            logging.info(
                'Train accuracy: %f' % current_train_accuracies_mean)

    return all_accuracies

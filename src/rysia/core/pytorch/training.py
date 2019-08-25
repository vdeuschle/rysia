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
import torch

from ...utils.misc import accuracy
from ..layer import Model
from .model import get_eval_model
from .helper import get_optimizer
from .layer_factory import bp2torch_loss


def stochastic_gradient_descent(model: Model,
                                ctx: str,
                                train_data: np.ndarray,
                                train_label: np.ndarray,
                                test_data: Optional[np.ndarray],
                                test_label: Optional[np.ndarray],
                                loss_func: str,
                                optimizer: str,
                                num_epochs: int,
                                learning_rate: float,
                                batch_size: Optional[int]) -> List[float]:
    train_label = train_label.argmax(axis=1) if loss_func == 'cross_entropy' else train_label
    train_data_size = train_data.shape[0]
    if batch_size is None or batch_size > train_data_size:
        batch_size = train_data_size
    loss_function = bp2torch_loss(loss_func)
    eval_model = get_eval_model(model)
    optimizer = get_optimizer(model, optimizer, learning_rate)
    all_accuracies = [['train accuracy']]

    testing = not (test_data is None or test_label is None)
    if testing:
        test_data = torch.tensor(test_data,
                                 device=ctx,
                                 dtype=torch.float)
        all_accuracies[0] += ['test accuracy']

    for i in range(num_epochs):
        current_train_accuracies = []
        perm = np.random.permutation(train_data_size)
        for j in range(0, train_data_size-train_data_size%batch_size, batch_size):
            idx = perm[j:j+batch_size]
            current_batch = train_data[idx]
            current_label = train_label[idx]
            train_prediction = eval_model(torch.tensor(current_batch,
                                                       device=ctx,
                                                       dtype=torch.float))[0]
            train_loss = loss_function(train_prediction, torch.tensor(current_label,
                                                                      device=ctx,
                                                                      dtype=torch.long))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            current_train_accuracies += [accuracy(train_prediction.cpu().detach().numpy(), current_label)]
        current_train_accuracies_mean = np.array(current_train_accuracies).mean()
        all_accuracies += [[current_train_accuracies_mean]]

        if testing:
            test_prediction = eval_model(test_data)[0]
            test_accuracy = accuracy(test_prediction.cpu().detach().numpy(), test_label)
            all_accuracies[-1] += [test_accuracy]
            logging.info(
                f'Train accuracy: {current_train_accuracies_mean}; Test accuracy: {test_accuracy}')
        else:
            logging.info(
                'Train accuracy: %f' % current_train_accuracies_mean)

    return all_accuracies


def recurrent_gradient_descent(model: Model,
                               ctx: str,
                               train_data: np.ndarray,
                               train_label: np.ndarray,
                               test_data: Optional[np.ndarray],
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
    train_label = train_label.argmax(axis=1) if loss_func == 'cross_entropy' else train_label
    train_data_size = train_data.shape[0]
    label_dim = train_label.shape[-1]
    if batch_size is None or batch_size > train_data_size:
        batch_size = train_data_size
    return_full_seq = len(train_data.shape) == len(train_label.shape)
    iterate_over_seq = train_seq_length != truncated_backprop_length
    loss_function = bp2torch_loss(loss_func)
    eval_model = get_eval_model(model)
    optimizer = get_optimizer(model, optimizer, learning_rate)
    all_accuracies = [['train accuracy']]

    testing = not (test_data is None or test_label is None)
    if testing:
        test_data_size = test_data.shape[0]
        test_data = torch.tensor(test_data,
                                 device=ctx,
                                 dtype=torch.float)
        init_test_states = [torch.tensor(np.zeros((test_data_size, state_size)),
                                         device=ctx,
                                         dtype=torch.float)
                            for state_size in state_sizes for _ in range(2)]
        all_accuracies[0] += ['test accuracy']

    for i in range(num_epochs):
        current_train_accuracies = []
        perm = np.random.permutation(train_data_size)
        for j in range(0, train_data_size-train_data_size%batch_size, batch_size):
            idx = perm[j:j+batch_size]
            current_batch = train_data[idx]
            current_label = train_label[idx]
            init_states = [torch.tensor(np.zeros((batch_size, state_size)),
                                        device=ctx,
                                        dtype=torch.float)
                           for state_size in state_sizes for _ in range(2)]
            for k in range(0, train_seq_length, truncated_backprop_length):
                batch_X = current_batch[:, k:k+truncated_backprop_length, :]
                batch_Y = current_label
                if return_full_seq:
                    batch_Y = batch_Y[:, k:k+truncated_backprop_length, :].reshape(-1, label_dim)
                train_prediction = eval_model(torch.tensor(batch_X,
                                                           device=ctx,
                                                           dtype=torch.float),
                                              truncated_backprop_length, init_states)
                train_loss = loss_function(train_prediction[0], torch.tensor(batch_Y,
                                                                             device=ctx,
                                                                             dtype=torch.long))
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                if iterate_over_seq:
                    init_states = list(map(lambda x: x.detach(), train_prediction[1:]))
                current_train_accuracies += [accuracy(train_prediction[0].cpu().detach().numpy(), batch_Y)]
        current_train_accuracies_mean = np.array(current_train_accuracies).mean()
        all_accuracies += [[current_train_accuracies_mean]]

        if testing:
            test_prediction = eval_model(test_data, test_seq_length, init_test_states)
            test_accuracy = accuracy(test_prediction[0].cpu().detach().numpy(), test_label)
            all_accuracies[-1] += [test_accuracy]
            logging.info(
                f'Train accuracy: {current_train_accuracies_mean}; Test accuracy: {test_accuracy}')
        else:
            logging.info(
                'Train accuracy: %f' % current_train_accuracies_mean)

    return all_accuracies

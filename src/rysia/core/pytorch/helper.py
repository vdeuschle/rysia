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

import torch
import numpy as np

from ..layer import Model
from .model import torch_model2list


def stack_sequences(input: torch.Tensor, seq_length: np.ndarray):
    input_list = torch.split(input, split_size_or_sections=seq_length.tolist())
    input_stacked = torch.nn.utils.rnn.pack_sequence(input_list)
    input_padded = torch.nn.utils.rnn.pad_packed_sequence(input_stacked, batch_first=True)
    return input_padded[0]


def unstack_sequences(input: torch.Tensor, seq_length: np.ndarray) -> torch.Tensor:
    first_seq = input[0]
    for current_seq, current_length in zip(input[1:], seq_length[1:]):
        first_seq = torch.cat((first_seq, current_seq[:current_length]), 0)
    return first_seq


def get_optimizer(model: Model,
                  optimizer: str,
                  learning_rate: float) -> torch.optim.Optimizer:
    if optimizer == 'sgd':
        return torch.optim.SGD(torch_model2list(model), lr=learning_rate)
    elif optimizer == 'adam':
        return torch.optim.Adam(torch_model2list(model), lr=learning_rate)
    else:
        raise RuntimeError(f'Unknown optimizer {optimizer} in get_train_step')

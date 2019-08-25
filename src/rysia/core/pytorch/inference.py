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

import torch
import torch.nn.functional as F
import numpy as np

from ..layer import Model
from .model import get_eval_model


def inference(model: Model,
              ctx: str,
              data: np.ndarray,
              batch_size: int) -> List[float]:
    data_size = data.shape[0]

    counter = 0
    eval_model = get_eval_model(model)
    with torch.no_grad():
        stop = time.time() + 60
        while time.time() < stop:
            idx = np.random.choice(data_size, batch_size)
            test_batch = data[idx]
            output = eval_model(torch.tensor(test_batch,
                                             device=ctx,
                                             dtype=torch.float))[0]
            output = F.softmax(output, dim=1).detach().numpy()
            counter += 1

    batches_per_second = counter / 60
    return [stop - 60, stop, batches_per_second]


def recurrent_inference(model: Model,
                        ctx: str,
                        data: np.ndarray,
                        batch_size: int,
                        state_sizes: List[int]) -> List[float]:
    data_size = data.shape[0]
    seq_length = data.shape[1]

    counter = 0
    eval_model = get_eval_model(model)
    with torch.no_grad():
        stop = time.time() + 60
        while time.time() < stop:
            idx = np.random.choice(data_size, batch_size)
            test_batch = data[idx]
            init_states = [torch.tensor(np.zeros((batch_size, state_size)),
                                        device=ctx,
                                        dtype=torch.float)
                           for state_size in state_sizes
                           for _ in range(2)]
            output = eval_model(torch.tensor(test_batch,
                                             device=ctx,
                                             dtype=torch.float),
                                seq_length,
                                init_states)[0]
            output = F.softmax(output, dim=1).detach().numpy()
            counter += 1

    batches_per_second = counter / 60
    return [stop - 60, stop, batches_per_second]

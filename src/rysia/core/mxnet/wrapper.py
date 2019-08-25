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
from pathlib import Path
from typing import Optional, List

import numpy as np
import mxnet as mx
import boto3

from ...utils.misc import create_csv_string
from .. import blueprint as bp
from .model import bp2mxnet_model
from .training import stochastic_gradient_descent, recurrent_gradient_descent
from .inference import inference, recurrent_inference


class MxnetModel:

    def __init__(self,
                 architecture: bp.Model,
                 params: Optional[List[np.ndarray]] = None) -> None:
        self.ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
        self.model = bp2mxnet_model(architecture)
        self.params = params

    def inference(self,
                  data: np.ndarray,
                  batch_size: int,
                  result_folder: Path,
                  mode: str,
                  optimizer_mode: str,
                  state_sizes: Optional[List[int]],
                  bucket_name: Optional[str] = None,
                  aws: bool = False) -> None:
        acc = [['start', 'end', 'batches per second']]
        if optimizer_mode == 'default':
            acc += [inference(self.model, self.ctx, self.params, data, batch_size)]
        elif optimizer_mode == 'recurrent':
            acc += [recurrent_inference(self.model, self.ctx, self.params, data, batch_size, state_sizes)]
        else:
            raise RuntimeError(f'unknown optimizer mode {optimizer_mode} in inference')

        csv_string = create_csv_string(acc)
        inference_filename = result_folder / f'{mode}.csv'

        if aws:
            s3_bucket = boto3.resource('s3').Bucket(bucket_name)
            s3_bucket.put_object(Key=str(inference_filename), Body=csv_string)
        else:
            with open(inference_filename, 'w') as output:
                output.write(csv_string)

    def training(self,
                 train_data: np.array,
                 train_label: np.ndarray,
                 test_data: Optional[np.ndarray],
                 test_label: Optional[np.ndarray],
                 loss_func: str, optimizer: str,
                 optimizer_mode: str,
                 num_epochs: int,
                 learning_rate: float,
                 result_folder: Path,
                 batch_size: Optional[int],
                 train_seq_length: Optional[int],
                 test_seq_length: Optional[int],
                 state_sizes: Optional[List[int]],
                 truncated_backprop_length: Optional[int],
                 bucket_name: Optional[str] = None,
                 aws: bool = False) -> None:
        runtime = [['start', 'end', 'runtime (sec)']]

        if optimizer_mode == 'default':
            start = time.time()
            accuracies = stochastic_gradient_descent(self.model,
                                                     self.ctx,
                                                     train_data,
                                                     train_label,
                                                     test_data,
                                                     test_label,
                                                     loss_func,
                                                     optimizer,
                                                     num_epochs,
                                                     learning_rate,
                                                     batch_size)
            end = time.time()
        elif optimizer_mode == 'recurrent':
            start = time.time()
            accuracies = recurrent_gradient_descent(self.model,
                                                    self.ctx,
                                                    train_data,
                                                    train_label,
                                                    test_data,
                                                    test_label,
                                                    train_seq_length,
                                                    test_seq_length,
                                                    state_sizes,
                                                    truncated_backprop_length,
                                                    loss_func,
                                                    optimizer,
                                                    num_epochs,
                                                    learning_rate,
                                                    batch_size)
            end = time.time()
        else:
            raise RuntimeError(f'Unknown optimizer mode {optimizer_mode} in fit')

        runtime += [[start, end, end-start]]
        runtime_csv_string = create_csv_string(runtime)
        runtime_filename = result_folder / 'runtime.csv'
        accuracies_csv_string = create_csv_string(accuracies)
        accuracies_filename = result_folder / 'accuracies.csv'

        if aws:
            s3_bucket = boto3.resource('s3').Bucket(bucket_name)
            s3_bucket.put_object(Key=str(runtime_filename), Body=runtime_csv_string)
            s3_bucket.put_object(Key=str(accuracies_filename), Body=accuracies_csv_string)
        else:
            with open(runtime_filename, 'w') as runtime_file, open(accuracies_filename, 'w') as accuracies_file:
                runtime_file.write(runtime_csv_string)
                accuracies_file.write(accuracies_csv_string)

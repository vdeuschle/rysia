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

import os
from typing import Union, List, Any, Optional
from functools import reduce
from pathlib import Path

import numpy as np


def create_csv_string(data: List[Any]) -> str:
    return '\n'.join(map(lambda x: ','.join(map(str, x)), data))


def accuracy(prediction: np.array, label: np.array) -> np.float64:
    prediction = prediction.argmax(axis=1) if len(prediction.shape) > 1 else prediction
    label = label.argmax(axis=1) if len(label.shape) > 1 else label
    return (prediction == label).mean()


def prod(value_list: List[Union[int, float]]) -> Union[int, float]:
    return reduce(lambda x,y: x*y, value_list)


def set_framework_seed(framework: str, seed: int) -> None:
    if framework == 'tensorflow':
        import tensorflow as tf
        tf.set_random_seed(seed)
    elif framework == 'pytorch':
        import torch
        torch.manual_seed(seed)
    elif framework == 'mxnet':
        import mxnet as mx
        mx.random.seed(seed)
    else:
        raise RuntimeError(f'Unsupported framework {framework} in set_framework_seed')


def create_blueprint_string(blueprint: str,
                            framework: str,
                            timestamp: str,
                            instance: Optional[str] = None) -> str:
    frameworks_start_idx = blueprint.find('frameworks')
    frameworks_end_idx = blueprint.find('\n', frameworks_start_idx)
    blueprint = f'{blueprint[:frameworks_start_idx]}framework = \'{framework}\'{blueprint[frameworks_end_idx:]}'
    if instance is not None:
        instances_start_idx = blueprint.find('instance_types')
        instances_end_idx = blueprint.find('\n', instances_start_idx)
        blueprint = f'{blueprint[:instances_start_idx]}instance_type = \'{instance}\'{blueprint[instances_end_idx:]}'
    return f'{blueprint}\ntimestamp = \'{timestamp}\''


def check_blueprints(root_folder_path: Path, **kwargs) -> Path:
    for root, dirs, file_names in os.walk(root_folder_path):
        for file_name in file_names:
            if file_name == 'bp.txt':
                with open(Path(root, file_name), 'r') as file:
                    text = file.read()
                for key, value in kwargs.items():
                    key += ' '
                    if isinstance(value, str):
                        value = f'\'{value}\''
                    start_idx = text.find(key)
                    end_idx = text.find('\n', start_idx)
                    field = text[start_idx:end_idx]
                    if not field.endswith(str(value)):
                        yield field, Path(root, file_name)


def select_blueprints(root_folder_path: Path, **kwargs) -> Path:
    for root, dirs, file_names in os.walk(root_folder_path):
        for file_name in file_names:
            if file_name == 'bp.txt':
                with open(Path(root, file_name), 'r') as file:
                    text = file.read()
                select = True
                for key, value in kwargs.items():
                    if isinstance(value, str):
                        value = f'\'{value}\''
                    start_idx = text.find(f'{key} =')
                    end_idx = text.find('\n', start_idx)
                    field = text[start_idx:end_idx]
                    select = select and field.endswith(str(value))
                if select:
                    yield Path(root, file_name)


def get_padding(kernel_size: int, input_size: int) -> float:
    if input_size % kernel_size == 0:
        return 0
    else:
        return (kernel_size - (input_size % kernel_size)) / 2

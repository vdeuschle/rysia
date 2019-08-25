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

from pathlib import Path
from typing import Optional, List, Tuple, Union

import numpy as np

from ..core import blueprint as bm
from .monitor import CPUMonitor, MemoryMonitor, DiskIOMonitor, GPUMonitor


def get_monitor(monitor: str,
                result_folder: Path,
                bucket_name: str = None,
                aws: bool = False) -> Union[CPUMonitor, MemoryMonitor, DiskIOMonitor, GPUMonitor]:
    if monitor == 'cpu':
        return CPUMonitor(str(result_folder / 'cpu.csv'), aws, bucket_name)
    elif monitor == 'memory':
        return MemoryMonitor(str(result_folder / 'memory.csv'), aws, bucket_name)
    elif monitor == 'diskIO':
        return DiskIOMonitor(str(result_folder / 'diskIO.csv'), aws, bucket_name)
    elif monitor == 'gpu':
        return GPUMonitor(str(result_folder / 'gpu.csv'), aws, bucket_name)
    else:
        raise RuntimeError(f'Unsupported monitor {monitor} in get_monitor')


def get_model(model: bm.Model,
              framework: str,
              params: Optional[List[Tuple[np.ndarray]]] = None) -> Union['TensorflowModel',
                                                                         'PytorchModel',
                                                                         'MxnetModel']:
    if framework == 'tensorflow':
        from rysia.core.tensorflow.wrapper import TensorflowModel
        return TensorflowModel(model, params=params)
    elif framework == 'pytorch':
        from rysia.core.pytorch.wrapper import PytorchModel
        return PytorchModel(model, params=params)
    elif framework == 'mxnet':
        from rysia.core.mxnet.wrapper import MxnetModel
        return MxnetModel(model, params=params)
    else:
        raise RuntimeError(f'Unsupported framework {framework} in get_model')

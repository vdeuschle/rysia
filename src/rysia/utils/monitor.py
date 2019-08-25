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
from typing import Optional

import psutil as ps
import boto3
from py3nvml.py3nvml import *

from ..utils.misc import create_csv_string


class Monitor(threading.Thread):
    def __init__(self,
                 filename: str,
                 aws: bool,
                 bucketname: str) -> None:
        super(Monitor, self).__init__()
        self.running = False
        self.aws = aws
        if self.aws:
            self.s3_bucket = boto3.resource('s3').Bucket(bucketname)
        self.filename = filename

    def __enter__(self) -> None:
        self.start()

    def __exit__(self, *args) -> None:
        self.running = False
        self.join()

    def run(self) -> None:
        self.running = True
        self.acc = []
        try:
            self.initialize()
            while self.running:
                time.sleep(1)
                self.execute()
        except (NVMLError, AttributeError, OSError) as e:
            self.acc += [[f'Monitor {self.__class__.__name__} failed to execute']]
            self.acc += [[str(e)]]
        finally:
            self.finalize()

    def initialize(self) -> None:
        raise NotImplementedError

    def execute(self) -> None:
        raise NotImplementedError

    def finalize(self) -> None:
        csv_string = create_csv_string(self.acc)
        if self.aws:
            self.s3_bucket.put_object(Key=self.filename, Body=csv_string)
        else:
            with open(self.filename, "w") as output:
                output.write(csv_string)


class CPUMonitor(Monitor):
    def __init__(self,
                 filename: str,
                 aws: bool = False,
                 bucketname: Optional[str] = None) -> None:
        super(CPUMonitor, self).__init__(filename, aws, bucketname)

    def initialize(self) -> None:
        cpu_count = ps.cpu_count()
        percent_label = [f'cpu{i}_percent' for i in range(cpu_count)]
        self.acc += [['timestamp',
                      *percent_label,
                      'user_time',
                      'system_time']]

    def execute(self) -> None:
        timestamp = time.time()
        cpu_percent = ps.cpu_percent(percpu=True)
        cpu_times = ps.cpu_times()
        self.acc += [[timestamp,
                      *cpu_percent,
                      cpu_times[0],
                      cpu_times[1]]]


class MemoryMonitor(Monitor):
    def __init__(self,
                 filename: str,
                 aws: bool = False,
                 bucketname: Optional[str] = None) -> None:
        super(MemoryMonitor, self).__init__(filename, aws, bucketname)

    def initialize(self) -> None:
        self.acc += [['timestamp',
                      'total_memory',
                      'available_memory',
                      'used_memory']]

    def execute(self) -> None:
        timestamp = time.time()
        memory_info = ps.virtual_memory()
        self.acc += [[timestamp,
                      memory_info.total,
                      memory_info.available,
                      memory_info.percent]]


class DiskIOMonitor(Monitor):
    def __init__(self,
                 filename: str,
                 aws: bool = False,
                 bucketname: Optional[str] = None) -> None:
        super(DiskIOMonitor, self).__init__(filename, aws, bucketname)

    def initialize(self) -> None:
        self.acc += [['timestamp',
                      'read_count',
                      'write_count',
                      'read_bytes',
                      'write_bytes']]

    def execute(self) -> None:
        timestamp = time.time()
        diskIO = ps.disk_io_counters()
        self.acc += [[timestamp,
                      diskIO.read_count,
                      diskIO.write_count,
                      diskIO.read_bytes,
                      diskIO.write_bytes]]


class GPUMonitor(Monitor):
    def __init__(self,
                 filename: str,
                 aws: bool = False,
                 bucketname: Optional[str] = None) -> None:
        super(GPUMonitor, self).__init__(filename, aws, bucketname)

    def initialize(self) -> None:
        self.acc += [['timestamp',
                      'id',
                      'memory_used',
                      'memory_total',
                      'memory_util_rate',
                      'gpu_util_rate']]
        nvmlInit()
        self.deviceCount = nvmlDeviceGetCount()

    def execute(self) -> None:
        timestamp = time.time()
        for id in range(self.deviceCount):
            handle = nvmlDeviceGetHandleByIndex(id)
            memory_info = nvmlDeviceGetMemoryInfo(handle)
            gpu_info = nvmlDeviceGetUtilizationRates(handle)
            memory_used = memory_info.used
            memory_total = memory_info.total
            memory_util_rate = gpu_info.memory
            gpu_util_rate = gpu_info.gpu
            self.acc += [[timestamp,
                          id,
                          memory_used,
                          memory_total,
                          memory_util_rate,
                          gpu_util_rate]]

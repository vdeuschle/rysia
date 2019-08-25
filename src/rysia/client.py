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

import random
import os
import logging
import pickle
from pathlib import Path
from datetime import datetime
from threading import Thread
from contextlib import ExitStack

import numpy as np

from .utils.misc import create_blueprint_string, set_framework_seed
from .utils.factory import get_model, get_monitor
from .utils.aws import (EnvironmentManager,
                        s3_upload_blueprint,
                        s3_download_results,
                        register_and_execute_job)
from .utils.ec2_instances import instances


def run_on_aws(bp,
               framework: str,
               instance: str) -> Thread:
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    instance_list = instance.split('.')

    result_folder_path = Path(bp.job_name, framework, *instance_list)
    local_result_folder_path = Path(bp.result_folder_path) / result_folder_path
    s3_result_folder_path = Path('dl-benchmarking', 'results') / result_folder_path

    try:
        experiment_idx = max(map(int, os.listdir(local_result_folder_path))) + 1
    except (FileNotFoundError, ValueError):
        experiment_idx = 1
    finally:
        local_result_folder_path /= str(experiment_idx)
        s3_result_folder_path /= str(experiment_idx)
        os.makedirs(local_result_folder_path)
        logging.info(f'Local result folder {local_result_folder_path} created')

    bp_file_name = Path(bp.__file__).parts[-1]
    with open(bp.__file__, 'r') as bp_input, open(local_result_folder_path / bp_file_name, 'w') as bp_output:
        blueprint_string = create_blueprint_string(bp_input.read(),
                                                   framework,
                                                   timestamp,
                                                   instance)
        bp_output.write(blueprint_string)

    env_name = instance.replace('.', '_')

    if instance.startswith('p'):
        try:
            container_image = bp.container_images[1]
            job_def_name = bp.job_def_names[1]
        except IndexError:
            container_image = bp.container_images[0]
            job_def_name = bp.job_def_names[0]
    else:
        container_image = bp.container_images[0]
        job_def_name = bp.job_def_names[0]

    job_name = f'{bp.job_name}_{framework}_{env_name}_{experiment_idx}'

    job_num_vcpu = bp.job_num_vcpu or instances[instance]['cpu']
    job_memory_size = bp.job_memory_size or instances[instance]['memory'] - 2000

    s3_upload_blueprint(bp.bucket_name,
                        s3_result_folder_path,
                        blueprint_string)

    def run():
        env_manager = EnvironmentManager(bp.region_name,
                                         env_name, instance,
                                         bp.ami_id,
                                         bp.env_min_cpu,
                                         bp.env_desired_cpu,
                                         bp.env_max_cpu,
                                         bp.subnets,
                                         bp.security_group_ids,
                                         bp.instance_role,
                                         bp.service_role,
                                         job_name,
                                         bp.tear_down_comp_env)

        with env_manager:
            job_succesfull = register_and_execute_job(bp.region_name,
                                                      job_name,
                                                      job_def_name,
                                                      env_name,
                                                      bp.bucket_name,
                                                      s3_result_folder_path,
                                                      bp.account_id,
                                                      container_image,
                                                      job_num_vcpu,
                                                      job_memory_size)

        if job_succesfull:
            s3_download_results(bp.bucket_name,
                                local_result_folder_path,
                                s3_result_folder_path)

    return Thread(target=run, name=job_name)


def run_locally(bp, framework: str) -> None:
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    result_folder_path = Path(bp.result_folder_path, bp.job_name, framework, 'local')

    try:
        experiment_idx = max(map(int, os.listdir(result_folder_path))) + 1
    except (FileNotFoundError, ValueError):
        experiment_idx = 1
    finally:
        result_folder_path /= str(experiment_idx)
        os.makedirs(result_folder_path)
        logging.info(f'Local result folder {result_folder_path} created')

    bp_file_name = Path(bp.__file__).parts[-1]
    with open(bp.__file__, 'r') as bp_input, open(result_folder_path / bp_file_name, 'w') as bp_output:
        blueprint_string = create_blueprint_string(bp_input.read(), framework, timestamp)
        bp_output.write(blueprint_string)

    random.seed(bp.python_seed)
    np.random.seed(bp.numpy_seed)
    set_framework_seed(framework, bp.framework_seed)

    train_data = np.load(Path(bp.train_data_path)).reshape(*bp.reshape_input_data)
    train_label = np.load(Path(bp.train_label_path))

    try:
        test_data = np.load(Path(bp.test_data_path)).reshape(*bp.reshape_input_data)
        test_label = np.load(Path(bp.test_label_path))
    except (TypeError, AttributeError):
        test_data = None
        test_label = None

    try:
        with open(bp.model_params_path, 'rb') as f:
            params = pickle.load(f)
    except (TypeError, FileNotFoundError):
        params = None

    if framework in ['pytorch', 'mxnet']:  # Use NCHW format for image data in pytorch and mxnet
        if len(train_data.shape) > 3:
            train_data = train_data.swapaxes(-3, -1)
            if isinstance(test_data, np.ndarray):
                test_data = test_data.swapaxes(-3, -1)

    model = get_model(bp.architecture, framework, params)

    if bp.mode == 'training':
        with ExitStack() as stack:
            for monitor in bp.monitors:
                stack.enter_context(get_monitor(monitor, result_folder_path))
            model.training(train_data,
                           train_label,
                           test_data,
                           test_label,
                           bp.loss_function,
                           bp.optimizer,
                           bp.optimizer_mode,
                           bp.epochs,
                           bp.learning_rate,
                           result_folder_path,
                           bp.batch_size,
                           bp.train_seq_length,
                           bp.test_seq_length,
                           bp.state_sizes,
                           bp.truncated_backprop_length)

    elif bp.mode == 'inference':
        with ExitStack() as stack:
            for monitor in bp.monitors:
                stack.enter_context(get_monitor(monitor, result_folder_path))
            model.inference(train_data,
                            bp.batch_size,
                            result_folder_path,
                            bp.mode,
                            bp.optimizer_mode,
                            bp.state_sizes)

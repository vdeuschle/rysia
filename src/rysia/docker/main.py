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
import logging
import os
import importlib.util
from pathlib import Path
from contextlib import ExitStack

import numpy as np
import boto3
from botocore.exceptions import ClientError

from ..utils.factory import get_monitor, get_model
from ..utils.aws import s3_fetch_data
from ..utils.misc import set_framework_seed


logging.basicConfig(level=logging.INFO)


def main() -> None:
    s3_path = Path(os.environ['S3_PATH'])
    bucket_name = os.environ['BUCKETNAME']
    blueprint_path = Path('home', 'dl-benchmarking', 'bp.py')

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    bucket.download_file(str(s3_path / 'bp.py'), str(blueprint_path))

    spec = importlib.util.spec_from_file_location('bp', blueprint_path)
    bp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bp)

    logging.info(f'{bp.framework}_{bp.job_name}_{bp.instance_type}')

    random.seed(bp.python_seed)
    np.random.seed(bp.numpy_seed)
    set_framework_seed(bp.framework, bp.framework_seed)

    train_data = s3_fetch_data(bp.bucket_name,
                               Path(bp.train_data_path)).reshape(*bp.reshape_input_data)
    train_label = s3_fetch_data(bp.bucket_name,
                                Path(bp.train_label_path))

    try:
        test_data = s3_fetch_data(bp.bucket_name,
                                  Path(bp.test_data_path)).reshape(*bp.reshape_input_data)
        test_label = s3_fetch_data(bp.bucket_name,
                                   Path(bp.test_label_path))
    except (TypeError, ClientError):
        test_data = None
        test_label = None

    try:
        params = s3_fetch_data(bucket_name, Path(bp.model_params_path))
    except (TypeError, ClientError):
        params = None

    if bp.framework in ['pytorch', 'mxnet']: # Use NCHW format for image data in pytorch and mxnet
        if len(train_data.shape) > 3:
            train_data = train_data.swapaxes(-3, -1)
            if isinstance(test_data, np.ndarray):
                test_data = test_data.swapaxes(-3, -1)

    model = get_model(bp.architecture, bp.framework, params)

    if bp.mode == 'training':
        with ExitStack() as stack:
            for monitor in bp.monitors:
                stack.enter_context(get_monitor(monitor, s3_path, bp.bucket_name, aws=True))
            model.training(train_data,
                           train_label,
                           test_data,
                           test_label,
                           bp.loss_function,
                           bp.optimizer,
                           bp.optimizer_mode,
                           bp.epochs,
                           bp.learning_rate,
                           s3_path,
                           bp.batch_size,
                           bp.train_seq_length,
                           bp.test_seq_length,
                           bp.state_sizes,
                           bp.truncated_backprop_length,
                           bp.bucket_name,
                           aws=True)

    elif bp.mode == 'inference':
        with ExitStack() as stack:
            for monitor in bp.monitors:
                stack.enter_context(get_monitor(monitor, s3_path, bp.bucket_name, aws=True))
            model.inference(train_data,
                            bp.batch_size,
                            s3_path,
                            bp.mode,
                            bp.optimizer_mode,
                            bp.state_sizes,
                            bp.bucket_name,
                            aws=True)


if __name__ == "__main__":
    main()

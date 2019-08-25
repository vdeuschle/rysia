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


import argparse
import importlib.util
import itertools
import logging
import time
import os

from . import client


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('botocore').setLevel(logging.CRITICAL)
logging.getLogger('boto3').setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s::%(asctime)s::%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('blueprint_path', help='path to blueprint file')
    args = parser.parse_args()
    blueprint_path = args.blueprint_path

    spec = importlib.util.spec_from_file_location('bp', blueprint_path)
    bp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bp)

    if bp.aws:
        threads = []

        for _, framework, instance in itertools.product(range(bp.runs), bp.frameworks, bp.instance_types):
            current_thread = client.run_on_aws(bp, framework, instance)
            threads += [current_thread]

        for current_thread in threads:
            current_thread.start()
            time.sleep(1)

        for current_thread in threads:
            current_thread.join()

    else:
        for _, framework in itertools.product(range(bp.runs), bp.frameworks):
            client.run_locally(bp, framework)


if __name__ == '__main__':
    main()

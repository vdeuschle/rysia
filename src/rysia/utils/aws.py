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
from typing import List
import os
import pickle
import logging

import numpy as np
import boto3
from botocore.exceptions import (ClientError,
                                 EndpointConnectionError,
                                 ReadTimeoutError)


def s3_upload_blueprint(bucket_name: str,
                        result_folder_path: Path,
                        blueprint: str) -> None:
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    blueprint_path = result_folder_path / 'bp.py'
    bucket.put_object(Key=str(blueprint_path), Body=blueprint)
    logging.info(f'Blueprint uploaded to bucket {bucket_name} at {blueprint_path}')


def s3_download_results(bucket_name: str,
                        local_result_folder:
                        Path, s3_result_folder: Path) -> None:
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=str(s3_result_folder)):
        key = obj.key
        if key.endswith('.csv'):
            metric = key.split('/')[-1]
            bucket.download_file(str(s3_result_folder / metric), str(local_result_folder / metric))
            logging.info(f'{metric} metrics downloaded to {local_result_folder}')


def s3_fetch_data(bucket_name: str, data_path: Path) -> np.ndarray:
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    bucket.download_file(str(data_path), 'temp')
    try:
        data = np.load('temp')
    except OSError:
        with open('temp', 'rb') as file:
            data = pickle.load(file)
    os.remove('temp')
    return data


def batch_create_compute_environment(region_name: str,
                                     env_name: str,
                                     instance_type: str,
                                     image_id: str,
                                     subnets: List[str],
                                     security_group_ids: List[str],
                                     instance_role: str,
                                     service_role: str,
                                     minv_cpus: int,
                                     desiredv_cpus: int,
                                     maxv_cpus: int,
                                     job_name: str,
                                     type: str = 'MANAGED',
                                     comp_env_state: str = 'ENABLED',
                                     resurce_type: str = 'EC2') -> None:
    client = boto3.client('batch', region_name=region_name)
    response = client.describe_compute_environments(computeEnvironments=[env_name])
    comp_env_exists = len(response['computeEnvironments'])
    if comp_env_exists:
        logging.info(
            f'Job {job_name} uses compute environment {env_name} with instance type {instance_type}')
    else:
        compute_resources = {'type': resurce_type,
                             'minvCpus': minv_cpus,
                             'maxvCpus': maxv_cpus,
                             'desiredvCpus': desiredv_cpus,
                             'instanceTypes': [instance_type],
                             'subnets': subnets,
                             'securityGroupIds': security_group_ids,
                             'instanceRole': instance_role}
        if image_id is not None:
            compute_resources['imageId'] = image_id
        client.create_compute_environment(computeEnvironmentName=env_name,
                                          type=type,
                                          state=comp_env_state,
                                          computeResources=compute_resources,
                                          serviceRole=service_role)
        comp_env_state = 'DISABLED'
        while comp_env_state != 'ENABLED':
            try:
                response = client.describe_compute_environments(computeEnvironments=[env_name])
                comp_env_state = response['computeEnvironments'][0]['state']
            except (ClientError, EndpointConnectionError, ReadTimeoutError):
                continue
            time.sleep(5)
        logging.info(
            f'Job {job_name} creates compute environment {env_name} with instance type {instance_type}')


def batch_create_job_queue(region_name: str,
                           queue_name: str,
                           job_name: str,
                           job_queue_state: str = 'ENABLED',
                           priority: int = 1) -> None:
    client = boto3.client('batch', region_name=region_name)
    compute_environment_order = [{'computeEnvironment': queue_name, 'order': 1}]
    response = client.describe_job_queues(jobQueues=[queue_name])
    job_queue_exists = len(response['jobQueues'])
    if job_queue_exists:
        logging.info(f'Job {job_name} uses job queue {queue_name}')
    else:
        client.create_job_queue(jobQueueName=queue_name,
                                priority=priority,
                                state=job_queue_state,
                                computeEnvironmentOrder=compute_environment_order)
        job_queue_state = 'DISABLED'
        while job_queue_state != 'ENABLED':
            try:
                response = client.describe_job_queues(jobQueues=[queue_name])
                job_queue_state = response['jobQueues'][0]['state']
            except (ClientError, EndpointConnectionError, ReadTimeoutError):
                continue
            time.sleep(5)
        logging.info(f'Job {job_name} creates job queue {queue_name}')


def register_and_execute_job(region_name: str,
                             job_name: str,
                             job_def_name: str,
                             queue_name: str,
                             bucket_name: str,
                             result_folder_path: Path,
                             account_id: str,
                             container_image: str,
                             vcpus: int,
                             memory: int) -> bool:
    client = boto3.client('batch', region_name=region_name)
    image = f'{account_id}.dkr.ecr.{region_name}.amazonaws.com/{container_image}'
    response = client.describe_job_definitions(jobDefinitionName=job_def_name)
    job_definition = response['jobDefinitions']
    for version in job_definition:
        if version['status'] == 'ACTIVE':
            revision = version['revision']
            logging.info(f'Job {job_name} uses job definition {job_def_name}:{revision}')
            break
    else:
        container_properties = {'image': image, 'vcpus': vcpus, 'memory': memory}
        job_definition = client.register_job_definition(jobDefinitionName=job_def_name,
                                                        type='container',
                                                        containerProperties=container_properties)
        revision = job_definition['revision']
        logging.info(f'Job {job_name} registeres job definition {job_def_name}:{revision}')
    container_override = {'environment': [{'name': 'BUCKETNAME', 'value': bucket_name},
                                          {'name': 'S3_PATH', 'value': str(result_folder_path)}],
                          'vcpus': vcpus,
                          'memory': memory}
    job = client.submit_job(jobName=job_name,
                            jobDefinition=f'{job_def_name}:{revision}',
                            jobQueue=queue_name,
                            containerOverrides=container_override)
    job_id = job['jobId']
    job_status = 'SUBMITTED'
    while job_status != 'SUCCEEDED' and job_status != 'FAILED':
        try:
            response = client.describe_jobs(jobs=[job_id])
            new_job_status = response['jobs'][0]['status']
        except (ClientError, EndpointConnectionError, ReadTimeoutError):
            continue
        if job_status != new_job_status:
            logging.info(f'Job {job_name} in status {new_job_status}')
        job_status = new_job_status
        time.sleep(5)
    return job_status == 'SUCCEEDED'


def batch_disable_and_delete_job_queue(region_name: str, queue_name :str) -> None:
    client = boto3.client('batch', region_name=region_name)
    response = client.describe_job_queues(jobQueues=[queue_name])
    job_queue_exists = len(response['jobQueues'])
    if job_queue_exists:
        client.update_job_queue(jobQueue=queue_name, state='DISABLED')
        job_queue_state = 'ENABLED'
        while job_queue_state != 'DISABLED':
            try:
                response = client.describe_job_queues(jobQueues=[queue_name])
                job_queue_state = response['jobQueues'][0]['state']
            except (ClientError, EndpointConnectionError):
                continue
            time.sleep(5)
        logging.info(f'Job Queue {queue_name} disabled')
        client.delete_job_queue(jobQueue=queue_name)
        job_queue_exists = 1
        while job_queue_exists:
            try:
                response = client.describe_job_queues(jobQueues=[queue_name])
                job_queue_exists = len(response['jobQueues'])
            except (ClientError, EndpointConnectionError, ReadTimeoutError):
                continue
            time.sleep(5)
        logging.info(f'Job Queue {queue_name} deleted')


def batch_disable_and_delete_compute_environment(region_name: str, env_name :str) -> None:
    client = boto3.client('batch', region_name=region_name)
    response = client.describe_compute_environments(computeEnvironments=[env_name])
    comp_env_exists = len(response['computeEnvironments'])
    if comp_env_exists:
        client.update_compute_environment(computeEnvironment=env_name, state='DISABLED')
        comp_env_state = 'ENABLED'
        while comp_env_state != 'DISABLED':
            try:
                response = client.describe_compute_environments(computeEnvironments=[env_name])
                comp_env_state = response['computeEnvironments'][0]['state']
            except (ClientError, EndpointConnectionError):
                continue
            time.sleep(5)
        logging.info(f'Compute Environment {env_name} disabled')
        client.delete_compute_environment(computeEnvironment=env_name)
        comp_env_exists = 1
        while comp_env_exists:
            try:
                response = client.describe_compute_environments(computeEnvironments=[env_name])
                comp_env_exists = len(response['computeEnvironments'])
            except (ClientError, EndpointConnectionError, ReadTimeoutError):
                continue
            time.sleep(5)
        logging.info(f'Compute Environment {env_name} deleted')


class EnvironmentManager:
    def __init__(self,
                 region_name: str,
                 env_name: str,
                 instance_type: str,
                 ami_id: str,
                 minv_cpus: int,
                 desiredv_cpus: int,
                 maxv_cpus: int,
                 subnets: List[str],
                 security_group_ids: List[str],
                 instance_role: str,
                 service_role: str,
                 job_name: str,
                 tear_down_comp_env: bool) -> None:
        self.region_name = region_name
        self.env_name = env_name
        self.instance_type = instance_type
        self.ami_id = ami_id
        self.env_minv_cpus = minv_cpus
        self.env_desiredv_cpus = desiredv_cpus
        self.env_maxv_cpus = maxv_cpus
        self.subnets = subnets
        self.security_group_ids = security_group_ids
        self.instance_role = instance_role
        self.service_role = service_role
        self.job_name = job_name
        self.tear_down_comp_env = tear_down_comp_env

    def __enter__(self) -> None:
        batch_create_compute_environment(self.region_name,
                                         self.env_name,
                                         self.instance_type,
                                         self.ami_id,
                                         self.subnets,
                                         self.security_group_ids,
                                         self.instance_role,
                                         self.service_role,
                                         self.env_minv_cpus,
                                         self.env_desiredv_cpus,
                                         self.env_maxv_cpus,
                                         self.job_name)
        batch_create_job_queue(self.region_name,
                               self.env_name,
                               self.job_name)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.tear_down_comp_env:
            batch_disable_and_delete_job_queue(self.region_name, self.env_name)
            batch_disable_and_delete_compute_environment(self.region_name, self.env_name)

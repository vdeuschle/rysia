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

import tensorflow as tf
import numpy as np

from ..layer import Model
from .model import get_eval_model


def inference(model: Model,
              ctx: str,
              data: np.ndarray,
              batch_size: int) -> List[float]:
    data_size = data.shape[0]
    input_dim = data.shape[1:]

    with tf.device(ctx):
        X = tf.placeholder(tf.float32, shape=[None, *input_dim])
        eval_model = get_eval_model(model)
        prediction = eval_model(X)[0]
        prediction = tf.nn.softmax(prediction)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        counter = 0
        stop = time.time() + 60
        while time.time() < stop:
            idx = np.random.choice(data_size, batch_size)
            test_batch = data[idx]
            output = sess.run(prediction, feed_dict={X: test_batch})
            counter += 1

    bps = counter / 60
    return [stop - 60, stop, bps]


def recurrent_inference(model: Model,
                        ctx: str,
                        data: np.ndarray,
                        batch_size: int,
                        state_sizes: List[int]) -> List[float]:
    data_size = data.shape[0]
    seq_length = data.shape[1]
    input_dim = data.shape[1:]

    with tf.device(ctx):
        X = tf.placeholder(tf.float32, shape=[None, *input_dim])
        states = [tf.placeholder(tf.float32, [None, state_size])
                  for state_size in state_sizes
                  for _ in range(2)]
        eval_model = get_eval_model(model)
        prediction = eval_model(X, seq_length, states)[0]
        prediction = tf.nn.softmax(prediction)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        counter = 0
        stop = time.time() + 60
        while time.time() < stop:
            idx = np.random.choice(data_size, batch_size)
            test_batch = data[idx]
            init_states = [np.zeros((test_batch.shape[0], state_size))
                           for state_size in state_sizes
                           for _ in range(2)]
            feed_dict = {X: test_batch, **{state: init_state
                                           for (state, init_state) in zip(states, init_states)}}
            output = sess.run(prediction, feed_dict=feed_dict)
            counter += 1

    bps = counter / 60
    return [stop - 60, stop, bps]

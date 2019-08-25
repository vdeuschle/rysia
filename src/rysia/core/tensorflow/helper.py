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

import tensorflow as tf


#https://stackoverflow.com/questions/49409488/tensorflow-tensor-reshape-and-pad-with-zeros-at-the-end-of-some-rows?rq=1
def stack_sequences(input: tf.Tensor, seq_length: tf.Tensor) -> tf.Tensor:
    data_len = tf.shape(input)[0]
    out_dim0 = tf.size(seq_length)
    out_dim1 = tf.reduce_max(seq_length)
    out_dim2 = input.get_shape()[-1]
    start_idxs = tf.concat([tf.constant([0]), tf.cumsum(seq_length)], axis=0)[:-1]
    pads = tf.fill([out_dim0], out_dim1) - seq_length
    reconstruction_metadata = tf.stack([start_idxs, seq_length, pads], axis=1)
    reconstruction_data = tf.map_fn(lambda x: tf.concat([tf.range(x[0],x[0]+x[1]),
                                                         tf.fill([x[2]], data_len)],
                                                        axis=0),
                                    reconstruction_metadata)
    output = tf.gather(tf.concat([input, tf.zeros((1, out_dim2))], axis=0),
                       tf.reshape(reconstruction_data, [out_dim0*out_dim1]))
    output = tf.reshape(output, [out_dim0, out_dim1, out_dim2]) - tf.constant(1, dtype=tf.float32)
    return output


def unstack_sequences(input: tf.Tensor, seq_length: tf.Tensor) -> tf.Tensor:
    input_shape = tf.shape(input)
    num_seqs = input_shape[0]
    time_steps = input_shape[1]
    input_dim = input.get_shape()[-1]
    input = tf.reshape(input, [num_seqs*time_steps, input_dim])
    start_idx = tf.concat([[0], tf.cumsum(tf.fill([num_seqs], time_steps))], axis=0)[:-1]
    idx = tf.stack([start_idx, start_idx + seq_length, seq_length], axis=1)
    idx_ranges = tf.map_fn(lambda x: tf.concat([tf.ones([x[-2] - x[-3]]),
                                                tf.zeros(time_steps-x[-1])], axis=0), idx)
    idx_ranges = tf.reshape(idx_ranges, [-1])
    output = tf.boolean_mask(input, idx_ranges)
    return output


def get_optimizer(optimizer: str, learning_rate: float) -> tf.train.Optimizer:
    if optimizer == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer == 'adam':
        return tf.train.AdamOptimizer(learning_rate)
    else:
        raise RuntimeError(f'Unknown optimizer {optimizer} in get_train_step')
# Copyright (C) 2016-2018 Alibaba Group Holding Limited
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import xdl
import time

reader = xdl.DataReader("r1", # name of reader
                        paths=["./generated_data.txt"] * 10, # file paths
                        enable_state=False) # enable reader state

reader.epochs(1).threads(1).batch_size(1000).label_count(1)
reader.feature(name='sparse0', type=xdl.features.sparse, serialized=True)\
    .feature(name='dense0', type=xdl.features.dense, nvec=13)
reader.startup()

def train():
    batch = reader.read()
    sess = xdl.TrainSession()
    emb1 = xdl.embedding('emb1', batch['sparse0'], xdl.TruncatedNormal(stddev=0.001), 64, 168642, vtype='index')
    loss = model_top(batch['dense0'], [emb1], batch['label'])
    train_op = xdl.Adagrad(0.5).optimize()
    log_hook = xdl.LoggerHook(loss, "loss:{0}", 10)

    print("Starting time measurement")
    start = time.time()
    sess = xdl.TrainSession(hooks=[log_hook])
    while not sess.should_stop():
        sess.run(train_op)
    end = time.time()
    print("TOTAL TIME:!!!!!!!!!!!!!!!!!!!")
    print(end - start)

@xdl.tf_wrapper()
def model_top(deep, embeddings, labels):
    bfc1 = tf.layers.dense(
        deep, 512, kernel_initializer=tf.truncated_normal_initializer(
            stddev=0.001, dtype=tf.float32))
    bfc2 = tf.layers.dense(
        bfc1, 256, kernel_initializer=tf.truncated_normal_initializer(
            stddev=0.001, dtype=tf.float32))
    bfc3 = tf.layers.dense(
        bfc2, 64, kernel_initializer=tf.truncated_normal_initializer(
            stddev=0.001, dtype=tf.float32))
    input = tf.concat([bfc3] + embeddings, 1)
    fc1 = tf.layers.dense(
        input, 512, kernel_initializer=tf.truncated_normal_initializer(
            stddev=0.001, dtype=tf.float32))
    fc2 = tf.layers.dense(
        fc1, 512, kernel_initializer=tf.truncated_normal_initializer(
            stddev=0.001, dtype=tf.float32))
    fc3 = tf.layers.dense(
        fc2, 256, kernel_initializer=tf.truncated_normal_initializer(
            stddev=0.001, dtype=tf.float32))
    y = tf.layers.dense(
        fc3, 1, kernel_initializer=tf.truncated_normal_initializer(
            stddev=0.001, dtype=tf.float32))
    loss = tf.losses.sigmoid_cross_entropy(labels, y)
    return loss

train()        

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
import numpy

reader = xdl.DataReader("r1", # name of reader
                        paths=["./generated_data.txt", "./generated_data.txt", "./generated_data.txt",  "./generated_data.txt",  "./generated_data.txt"], # file paths
                        enable_state=False) # enable reader state

reader.epochs(10).threads(4).batch_size(10).label_count(1)
reader.feature(name='sparse0', type=xdl.features.sparse, serialized=True)
reader.startup()

def train():
    batch = reader.read()
    emb1 = xdl.embedding('emb1', batch['sparse0'], xdl.Constant(0.0), 128, 10000000, vtype='index')
    loss = model([emb1], batch['label'])
    train_op = xdl.SGD(0.5).optimize()
    log_hook = xdl.LoggerHook(loss, "loss:{0}", 10)
    sess = xdl.TrainSession(hooks=[log_hook])
    print("Starting time measurement")
    start = time.time()
    while not sess.should_stop():
        sess.run(train_op)
    end = time.time()
    print("TOTAL TIME:!!!!!!!!!!!!!!!!!!!")
    print(end - start)

@xdl.tf_wrapper()
def model(embeddings, labels):
    input = tf.concat(embeddings, 1)
    fc1 = tf.layers.dense(
        input, 128, kernel_initializer=tf.truncated_normal_initializer(
            stddev=0.001, dtype=tf.float32))
    y = tf.layers.dense(
        fc1, 1, kernel_initializer=tf.truncated_normal_initializer(
            stddev=0.001, dtype=tf.float32))
    loss = tf.losses.sigmoid_cross_entropy(labels, y)
    return loss

train()

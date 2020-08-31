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

DATA_FILE = "/xdl_training_samples/data.txt"
EMB_DIMENSION = 187767405
NUM_WORKERS = 9
CKPT = True
CKPT_INTERVAL = 60
PRINT_INTERVAL = 60

def train():
    reader = xdl.DataReader("r1", # name of reader
                            paths=[DATA_FILE] * NUM_WORKERS, # file paths
                            enable_state=False) # enable reader state

    reader.epochs(1).threads(16).batch_size(2048).label_count(1)
    reader.feature(name='sparse0', type=xdl.features.sparse, serialized=True) \
        .feature(name='dense0', type=xdl.features.dense, nvec=13)
    reader.startup()

    batch = reader.read()
    sess = xdl.TrainSession()
    #TODO: switch to uniform
    emb1 = xdl.embedding('emb1', batch['sparse0'], xdl.UniformUnitScaling(factor=0.125), 64, EMB_DIMENSION, vtype='index')
    loss = model_top(batch['dense0'], [emb1], batch['label'])
    train_op = xdl.SGD(0.1).optimize()

    print("Starting time measurement")
    start = time.time()
    sess = xdl.TrainSession()

    ckpt_mark = CKPT_INTERVAL
    print_mark = PRINT_INTERVAL
    ckpt_version = 0
    while not sess.should_stop():
        sess.run(train_op)
        cur_time = time.time()
        if cur_time - start > print_mark:
            print("Time into training: " + str(cur_time - start))
            print_mark += PRINT_INTERVAL
        if CKPT and int(xdl.get_task_index()) == 0:
            if cur_time - start > ckpt_mark:
                print("Taking ckpt {v} starting {s}".format(v=ckpt_version, s=cur_time - start))
                saver = xdl.Saver()
                saver.save(version = str(ckpt_version))
                after_time =time.time()
                print("Ending ckpt {v} starting {s}. Ckpt takes: {t}".format(v=ckpt_version, s=after_time - start, t=after_time-cur_time))
                ckpt_mark += CKPT_INTERVAL
                ckpt_version += 1

    end = time.time()
    print("TOTAL TIME:!!!!!!!!!!!!!!!!!!!")
    print(end - start)

@xdl.tf_wrapper()
def model_top(deep, embeddings, labels):
    def next_layer(prev, m, n):
        stddev = (2.0/(m+n))**0.5
        return tf.layers.dense(
            prev, n, kernel_initializer=tf.truncated_normal_initializer(
                stddev=stddev, dtype=tf.float32), activation=tf.nn.relu)
    #TODO: change all stddev
    bfc1 = next_layer(deep, 64, 512)
    bfc2 = next_layer(bfc1, 512, 256)
    bfc3 = next_layer(bfc2, 256, 64)
    input = tf.concat([bfc3] + embeddings, 1)
    fc1 =  next_layer(input, 128, 512)
    fc2 = next_layer(fc1, 512, 512)
    fc3 = next_layer(fc2, 512, 256)
    stddev = (2.0/(257))**0.5
    y = tf.layers.dense(
            fc3, 1, kernel_initializer=tf.truncated_normal_initializer(
                stddev=stddev, dtype=tf.float32))
    loss = tf.losses.sigmoid_cross_entropy(labels, y)
    return loss


if __name__ == "__main__":
    train()


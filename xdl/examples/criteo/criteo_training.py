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
import threading
import sys

DATA_FILE = "/xdl_training_samples/data.txt"
EMB_DIMENSION = 197767405
# EMB_DIMENSION = 19776740
NUM_COPIES = 297
CKPT = False
TOTAL_NUM_STEPS = 226004
TOTAL_NUM_CKPTS = 8
CKPT_INTERVAL_NUM_STEPS = TOTAL_NUM_STEPS/TOTAL_NUM_CKPTS

INITIAL_CKPT = False
BATCH_SIZE = 2048

TOT_STEP = 8000 # for mlc exps

step = 0
prev_step = 0
report_interval = 10
start = 0.0
report_count = 0

def my_print(x):
    print(x)
    sys.stdout.flush()


def report_step_change():
    global step
    global prev_step
    global report_interval
    global start
    global report_count
    while (time.time() - start) > report_interval * report_count:
        diff = step - prev_step
        prev_step = step
        my_print("({rc}, {diff}, {t}, {step})".format(diff=diff, rc=report_count, t=(time.time() - start), step=step))
        report_count += 1


def train():
    reader = xdl.DataReader("r1",  # name of reader
                            paths=[DATA_FILE] * NUM_COPIES,  # file paths
                            enable_state=False) # enable reader state

    reader.epochs(100).threads(32).batch_size(BATCH_SIZE).label_count(1)
    reader.feature(name='sparse0', type=xdl.features.sparse, serialized=True) \
        .feature(name='dense0', type=xdl.features.dense, nvec=13)
    reader.startup()

    batch = reader.read()
    #TODO: switch to uniform
    emb1 = xdl.embedding('emb1', batch['sparse0'], xdl.UniformUnitScaling(factor=0.125), 128, EMB_DIMENSION, vtype='index')
    loss = model_top(batch['dense0'], [emb1], batch['label'])
    train_op = xdl.SGD(0.1).optimize()

    my_print("Starting time measurement")

    global step
    global prev_step
    global report_interval
    global start

    start = time.time()
    sess = xdl.TrainSession()

    # run for one op first
    sess.run(train_op)

    if INITIAL_CKPT and  int(xdl.get_task_index()) == 0:
        cur_time = time.time()
        my_print("CKPT: Taking ckpt at step {s} starting {t}".format(s=step, t=cur_time - start))
        saver = xdl.Saver()
        saver.save(version = str(step))
        after_time =time.time()
        my_print("CKPT: Ending ckpt at step {s} starting {t}. Ckpt takes: {d}".format(s=step, t=after_time - start, d=after_time-cur_time))

    while not sess.should_stop():
        # if step >= TOTAL_NUM_STEPS:
        if step >= TOT_STEP:
            break
        sess.run(train_op)
        cur_time = time.time()
        report_step_change()

        step += 1

        # CKPT
        if CKPT and int(xdl.get_task_index()) == 1 and step % CKPT_INTERVAL_NUM_STEPS == 0:
            my_print("CKPT: Taking ckpt at step {s} starting {t}".format(s=step, t=cur_time - start))
            saver = xdl.Saver()
            saver.save(version = str(step))
            after_time =time.time()
            my_print("CKPT: Ending ckpt at step {s} starting {t}. Ckpt takes: {d}".format(s=step, t=after_time - start, d=after_time-cur_time))

    end = time.time()
    my_print("TOTAL TIME:!!!!!!!!!!!!!!!!!!!")
    my_print(end - start)

@xdl.tf_wrapper(device_type="gpu")
def model_top(deep, embeddings, labels):
    def next_layer(prev, m, n):
        stddev = (2.0/(m+n))**0.5
        return tf.layers.dense(
            prev, n, kernel_initializer=tf.truncated_normal_initializer(
                stddev=stddev, dtype=tf.float32), activation=tf.nn.relu)
    #TODO: change all stddev

    bfc1 = next_layer(deep, 64, 512)
    bfc2 = next_layer(bfc1, 512, 256)
    bfc3 = next_layer(bfc2, 256, 128)
    input = tf.concat([bfc3] + embeddings, 1)
    fc1 =  next_layer(input, 128, 1024)
    fc2 = next_layer(fc1, 1024, 1024)
    fc3 = next_layer(fc2, 1024, 512)
    fc4 = next_layer(fc3, 512, 256)

    stddev = (2.0/(257))**0.5
    y = tf.layers.dense(
        fc4, 1, kernel_initializer=tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32))
    loss = tf.losses.sigmoid_cross_entropy(labels, y)
    return loss


if __name__ == "__main__":
    train()

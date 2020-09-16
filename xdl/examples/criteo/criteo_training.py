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
EMB_DIMENSION = 187767405
NUM_WORKERS = 15
CKPT = False


# TODO: measure and change these two to match 4 hours and 1h/30min ckpt interval
TOTAL_NUM_STEPS = 62208
TOTAL_NUM_CKPTS = 4
CKPT_INTERVAL_NUM_STEPS = TOTAL_NUM_STEPS/TOTAL_NUM_CKPTS

step = 0
prev_step = 0
stop_report = False
report_interval = 10
start = 0.0

def my_print(x):
    print(x)
    sys.stdout.flush()


def report_step_change():
    report_count = 0
    global step
    global prev_step
    global stop_report
    global report_interval
    global start
    while not stop_report:
        diff = step - prev_step
        prev_step = step
        my_print("({rc}, {diff}, {t}, {step})".format(diff=diff, rc=report_count, t=(time.time() - start), step=step))
        target_next_print = report_interval * report_count
        time.sleep(max(target_next_print + start - time.time(), 0))
        report_count += 1


def train():
    reader = xdl.DataReader("r1", # name of reader
                            paths=[DATA_FILE] * NUM_WORKERS, # file paths
                            enable_state=False) # enable reader state

    reader.epochs(100).threads(16).batch_size(2048).label_count(1)
    reader.feature(name='sparse0', type=xdl.features.sparse, serialized=True) \
        .feature(name='dense0', type=xdl.features.dense, nvec=13)
    reader.startup()

    batch = reader.read()
    sess = xdl.TrainSession()
    #TODO: switch to uniform
    emb1 = xdl.embedding('emb1', batch['sparse0'], xdl.UniformUnitScaling(factor=0.125), 64, EMB_DIMENSION, vtype='index')
    loss = model_top(batch['dense0'], [emb1], batch['label'])
    train_op = xdl.SGD(0.1).optimize()

    my_print("Starting time measurement")

    global step
    global prev_step
    global stop_report
    global report_interval
    global start

    start = time.time()
    sess = xdl.TrainSession()

    # run for one op first
    sess.run(train_op)
    t = threading.Thread(target=report_step_change)
    t.start()

    while not sess.should_stop():
        if step >= TOTAL_NUM_STEPS:
            break
        sess.run(train_op)
        step += 1
        cur_time = time.time()

        # CKPT
        if CKPT and int(xdl.get_task_index()) and step % CKPT_INTERVAL_NUM_STEPS == 0:
            my_print("Taking ckpt {v} starting {s}".format(v=ckpt_version, s=cur_time - start))
            saver = xdl.Saver()
            saver.save(version = str(step))
            after_time =time.time()
            my_print("Ending ckpt {v} starting {s}. Ckpt takes: {t}".format(v=ckpt_version, s=after_time - start, t=after_time-cur_time))
    stop_report = True
    t.join()
    end = time.time()
    my_print("TOTAL TIME:!!!!!!!!!!!!!!!!!!!")
    my_print(end - start)

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

import json
import sys

import tensorflow as tf


@tf.function
def loss_function(real, predicted):
    vector = tf.losses.sparse_categorical_crossentropy(real, predicted)
    return tf.reduce_mean(vector)


@tf.function
def accuracy_function(real, predicted):
    vector = tf.metrics.sparse_categorical_accuracy(real, predicted)
    return tf.reduce_mean(vector)


def write_metrics(step, loss, accuracy, path, to_stdout=True):
    metrics_dict = dict(step=int(step),
                        loss=float(loss.numpy()),
                        accuracy=float(accuracy.numpy()))
    metrics_line = json.dumps(metrics_dict)

    with tf.io.gfile.GFile(path, mode="a+") as metrics_file:
        metrics_file.write(f"{metrics_line}\n")

    if to_stdout:
        tf.print(metrics_line, output_stream=sys.stdout)

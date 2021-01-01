import tensorflow as tf
from tensorflow.keras.layers import Dense


def _make_layer(units: int) -> tf.keras.layers.Layer:
    return Dense(units, activation=tf.keras.activations.relu,
                 kernel_initializer=tf.keras.initializers.he_normal(),
                 kernel_regularizer=tf.keras.regularizers.l2(),
                 bias_regularizer=tf.keras.regularizers.l2())
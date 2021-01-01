import sys
from typing import Tuple

import tensorflow as tf

from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.activations import softmax
from tensorflow.keras.models import Sequential

from mnist.models.utils import _make_layer


class GrayImageClassifier(tf.Module):

    # Constructor
    def __init__(self, layer_sizes: Tuple[int], image_height: int, image_width: int, number_of_classes: int, name=None):
        super(GrayImageClassifier, self).__init__(name)
        self._image_height = image_height
        self._image_width = image_width
        self._output_size = number_of_classes
        self._block = self._make_block(layer_sizes)

    # Batch functions
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 1], dtype=tf.dtypes.float32)])
    def __call__(self, batch: tf.Tensor):
        return self._block(batch, training=False)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 1], dtype=tf.dtypes.float32)])
    def train(self, batch):
        return self._block(batch, training=True)

    # Parameter functions
    @tf.function(input_signature=[])
    def image_height(self):
        return tf.constant(self._image_height, name="image_height")

    @tf.function(input_signature=[])
    def image_width(self):
        return tf.constant(self._image_width, name="image_width")

    @tf.function(input_signature=[])
    def output_size(self):
        return tf.constant(self._output_size, name="output_size")

    # Utils
    def _make_block(self, layers: Tuple[int]) -> tf.keras.Model:
        dense_layers = [_make_layer(size) for size in layers]

        input_layers = [Flatten()]
        output_layers = [Dense(self._output_size, activation=softmax)]

        layers = input_layers + dense_layers + output_layers
        model = Sequential(layers)

        model.build(input_shape=[None, self._image_height, self._image_width])

        return model

    # Model persistence functions
    def save(self, path: str):
        tf.print(f"Saving model to {path}", output_stream=sys.stdout)
        signatures = dict(serving_default=self.__call__,
                          image_height=self.image_height,
                          image_width=self.image_width,
                          output_size=self.output_size)

        tf.saved_model.save(self, path, signatures=signatures)

    @classmethod
    def load(cls, path: str):
        return tf.saved_model.load(path)

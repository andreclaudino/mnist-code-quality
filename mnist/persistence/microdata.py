from typing import Callable

import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod


def load_grayscale_images(dataset_path: str, batch_size: int, images_height: int,
                          images_width: int, epochs: int = 1) -> tf.data.Dataset:

    # list all files in folders subdirs
    paths_dataset = tf.data.Dataset.list_files(f"{dataset_path}/**/*", shuffle=True)

    # map each path to a tuple (path, class_name)
    paths_and_label_dataset = paths_dataset.map(_extract_label, num_parallel_calls=tf.data.AUTOTUNE)

    image_loader = _make_image_loader(images_height, images_width)
    images_dataset = paths_and_label_dataset.map(image_loader, num_parallel_calls=tf.data.AUTOTUNE)

    return images_dataset.repeat(epochs).prefetch(tf.data.AUTOTUNE).batch(batch_size).cache()


def _extract_label(path: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    class_name = tf.strings.split(path, "/")[-2]
    class_index = tf.strings.to_number(class_name, out_type=tf.dtypes.int32)
    return path, class_index


def _make_image_loader(images_height: int, images_width: int) -> Callable:
    image_dimensions = (images_height, images_width)

    def loader(path: tf.Tensor, label: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        # Load raw data from path
        raw = tf.io.read_file(path)

        # Decode image from raw data as a tensor of tf.float32
        image = tf.io.decode_image(raw, dtype=tf.float32, expand_animations=False, channels=1)

        # Resize the tensor image to the expected dimensions
        image = tf.image.resize(image, size=image_dimensions, antialias=True, preserve_aspect_ratio=False,
                                method=ResizeMethod.BILINEAR)
        image = _rescale(image)

        return image, label

    return loader


def _rescale(image: tf.Tensor) -> tf.Tensor:
    return image/255.0

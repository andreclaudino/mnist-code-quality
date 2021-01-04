from typing import Tuple

import click

import tensorflow as tf

from mnist.models.gray_image_classifier import GrayImageClassifier
from mnist.persistence.microdata import load_grayscale_images
from mnist.training.loop import run_training_loop


@click.command()
@click.option("--dataset-path", type=click.STRING, required=True, help="Path for the dataset used for training")
@click.option("--output-path", type=click.STRING, default="output",
              help="Path where checkpoints, metrics and model artifact will be saved")
@click.option("--batch-size", type=click.INT, default=32, help="Training batch size")
@click.option("--images-height", type=click.INT, required=True, help="Final height of images after resize")
@click.option("--images-width", type=click.INT, required=True, help="Final width of images after resize")
@click.option("--epochs", type=click.INT, default=1, help="Number of training epochs (repeats of dataset)")
@click.option("--learning-rate", type=click.FLOAT, required=True, help="Learning rate for gradient optimization")
@click.option("--debug/--no-debug", default=False, help="Whether or not to use tensorflow in debug mode")
@click.option("--layer-sizes", "_layer_size_string", type=click.STRING,
              help="Comma-separeted list of dense layer sizes for the model")
@click.option("--number-of-classes", type=click.INT,
              help="Number of output classes (the number os neurons in the output layer)")
@click.option("--summary-step-size", type=click.INT, default=10,
              help="Number of steps between each metric report and checkpoint save")
def main(dataset_path: str, output_path: str, batch_size: int, images_height: int, images_width: int,
         epochs: int, learning_rate: float, debug: bool, _layer_size_string: str, number_of_classes: int,
         summary_step_size: int):

    # Parse layer list
    layer_sizes = _parse_layer_sizes(_layer_size_string)

    # Setup debug environment if required
    tf.config.experimental_run_functions_eagerly(debug)

    # load train data
    dataset = load_grayscale_images(dataset_path, batch_size, images_height, images_width, epochs)

    # Create model
    model = GrayImageClassifier(layer_sizes, images_height, images_width, number_of_classes,
                                name="grayescale_classifier")

    # Training loop
    run_training_loop(model, dataset, learning_rate, summary_step_size, output_path)

    # Save trained model
    model.save(f"{output_path}/saved_model")


def _parse_layer_sizes(layer_sizes: str) -> Tuple[int]:
    try:
        return tuple([int(layer_size) for layer_size in layer_sizes.split(",")])
    except:
        raise ValueError("parameter layer-sizes should be a comma separated list of integers")


if __name__ == '__main__':
    main()

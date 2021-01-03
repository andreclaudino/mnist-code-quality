import click
import tensorflow as tf

from mnist.models.gray_image_classifier import GrayImageClassifier
from mnist.persistence.microdata import load_grayscale_images
from mnist.validating.loop import run_validating_loop


@click.command()
@click.option("--saved-model-path", type=click.STRING, required=True, help="Path to the saved model artifact")
@click.option("--dataset-path", type=click.STRING, required=True, help="Path to validation dataset")
@click.option("--batch-size", type=click.INT, default=32, help="Validation batch size")
def main(saved_model_path: str, dataset_path: str, batch_size: int):

    model: GrayImageClassifier = GrayImageClassifier.load(saved_model_path)

    images_height = model.image_height().numpy()
    images_width = model.image_width().numpy()

    dataset = load_grayscale_images(dataset_path, batch_size, images_height, images_width)

    run_validating_loop(model, dataset)


if __name__ == '__main__':
    main()

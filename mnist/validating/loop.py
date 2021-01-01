import tensorflow as tf

from mnist.metrics import loss_function, accuracy_function
from mnist.models.gray_image_classifier import GrayImageClassifier


def run_validating_loop(model: GrayImageClassifier, dataset: tf.data.Dataset):

    losses = []
    accuracies = []

    for step, (batch, labels) in enumerate(dataset):
        loss, accuracy = _run_test_step(model, batch, labels)
        print(f"batch {step}: loss: {loss}, accuracy: {accuracy}")

        losses.append(loss)
        accuracies.append(accuracy)

    mean_loss = tf.reduce_mean(losses)
    means_accuracy = tf.reduce_mean(accuracies)

    print(f"mean loss: {mean_loss}, mean accuracy: {means_accuracy}")


def _run_test_step(model: GrayImageClassifier, batch: tf.Tensor, labels: tf.Tensor):
    predicted = model(batch)

    loss = loss_function(labels, predicted)
    accuracy = accuracy_function(labels, predicted)

    return loss, accuracy

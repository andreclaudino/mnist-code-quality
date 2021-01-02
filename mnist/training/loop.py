import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2

from mnist.metrics import loss_function, write_metrics, accuracy_function
from mnist.models.gray_image_classifier import GrayImageClassifier
from mnist.traininig.checkpoints import make_checkpoint, save_checkpoint


def run_training_loop(model: GrayImageClassifier, dataset: tf.data.Dataset, learning_rate: float, summary_step_size: int,
                      outputs_path: str):

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    checkpoint_manager, checkpointer, prefix = make_checkpoint(model, optimizer, outputs_path)
    metrics_file = f"{outputs_path}/train_metrics.json"

    if checkpoint_manager.latest_checkpoint:
        print(f"Training restored from {checkpoint_manager.latest_checkpoint}")
    else:
        print(f"Training initialized from scratch")

    for step, (batch, labels) in enumerate(dataset):
        loss, accuracy = _run_training_step(model, optimizer, batch, labels)

        if step % summary_step_size == 0 and step > 0:
            write_metrics(step, loss, accuracy, metrics_file)
            save_checkpoint(checkpointer, checkpointer, prefix, loss, accuracy, step)


def _run_training_step(model: GrayImageClassifier, optimzer: OptimizerV2,
                       batch: tf.Tensor, labels: tf.Tensor) -> (float, float):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        predicted = model.train(batch)

        loss = loss_function(labels, predicted)
        accuracy = accuracy_function(labels, predicted)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimzer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, accuracy

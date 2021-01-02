import tensorflow as tf


def make_checkpoint(model, optimizer, save_path):
    # Create the checkpoint associating optimizer and model, and creating sthe tep and the metrics
    checkpoint_path = f"{save_path}/checkpoints"
    checkpointer = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model,
                                       loss=tf.Variable(0.0), accuracy=tf.Variable(0.0))

    # Create the checkpoint manage and configure it to keep at the most the last 3 saved
    checkpoint_manager = tf.train.CheckpointManager(checkpointer, checkpoint_path, max_to_keep=3)

    # Restore checkpoint or create a new if it doens't exist in the directory
    checkpointer.restore(checkpoint_manager.latest_checkpoint)

    return checkpoint_manager, checkpointer, f"{checkpoint_path}/step"


def save_checkpoint(checkpoint_manager, checkpoint_factory, prefix, loss, accuracy, step):
    # Assign the step and the metrics, loss and accuracy, to the checkpoint
    checkpoint_factory.step.assign(step)
    checkpoint_factory.loss.assign(loss)
    checkpoint_factory.accuracy.assign(accuracy)

    # Save the checkpoint
    saved_path = checkpoint_manager.save(file_prefix=prefix)
    tf.print(f"Checkpoint saved for step {step} at {saved_path}")

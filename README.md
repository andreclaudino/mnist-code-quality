# MNIST Code Quality

This is a Tensorflow implementation of a solution for MNIST problem writen on code quality principles. The problem itself is very easy, and can be resolved with simple solutions. The approach here is show how to organize an ML training and validating application, easily refactorable, organized by concerns.

## What is here

This project contains the following modules:

* `metrics`: There are the evaluation and training metrics used on inference.
* `models`: There is the model. This model corresponds to an artifact able to return all information needed for deployment.
* `persistence`: Functions to load and preprocess data for training and validating
* `training`: The training loop and entrypoint.
* `validating`: The validating loop and entrypoint.

## How to execute

First, create a virtual environment to avoid conflicts with other projects and system instalation.

```shell
$> git clone https://github.com/andreclaudino/mnist-code-quality.git
$> cd mnist-code-quality/
$> python3 -m virtualenv venv
$> source venv/bin/activate
$> pip3 install -e .
```

After instalation, there will be two commands, `train` (to train a new model) and `validate` (to validate a trained model). The usage of each can be foung using the `--help` option:

For **train**:

```shell
$> train --help
Usage: train [OPTIONS]

Options:
  --dataset-path TEXT          Path for the dataset used for training
                               [required]

  --output-path TEXT           path where checkpoints, metrics and model
                               artifact will be saved

  --batch-size INTEGER         Training batch size
  --images-height INTEGER      final height of images after resize  [required]
  --images-width INTEGER       Final width of images after resize  [required]
  --epochs INTEGER             Number of training epochs (repeats of dataset)
  --learning-rate FLOAT        Leargning rate for gradient optimization
                               [required]

  --debug / --no-debug         Should or not use tensorflow in debug mode
  --layer-sizes TEXT           Comma-separeted list of dense layer sizes for
                               the model

  --number-of-classes INTEGER  Number of output classes (the number os neurons
                               in the output layer)

  --summary-step-size INTEGER  Number of steps between each metric report and
                               checkpoint save

  --help                       Show this message and exit.

```

for **validate**:

```
$> validate --help
Usage: validate [OPTIONS]

Options:
  --saved-model-path TEXT  Path to the saved model artifact  [required]
  --dataset-path TEXT      Path to validating dataset  [required]
  --batch-size INTEGER     Validating batch size
  --help                   Show this message and exit.

```
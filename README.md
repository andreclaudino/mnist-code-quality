# MNIST Code Quality

\* Here is a [Brazilian Portuguese version](README.pt-br.md).  
\* Há uma versão em [Português Brasileiro](README.pt-br.md) neste link.

This is a Tensorflow implementation of a solution for the MNIST problem written according to code quality principles.
The problem itself is fairly easy, and can be resolved with simple solutions. The idea here is show how to organize an
ML training and validation application in such a way that it is easily refactorable and organized by responsibilities.

## Contents

This project contains the following modules:

* `metrics`: Contains the metrics used during training and validation;
* `models`: Contains the model. This model corresponds to an artifact able to return all information needed for deployment;
* `persistence`: Contains functions to load and preprocess data for training and validation;
* `training`: Contains the training entry point and loop;
* `validating`: Contains the validation entry point and loop.

## How to execute

First, create a virtual environment to avoid conflicts with other projects and the system installation. Then install the
requirements using `pip`.

```shell
$> git clone https://github.com/andreclaudino/mnist-code-quality.git
$> cd mnist-code-quality/
$> python3 -m virtualenv venv
$> source venv/bin/activate
$> pip3 install -e .
```

After instalation, two commands will be available: `train` (to train a new model) and `validate` (to validate a trained
model). The arguments for each command can be found using the `--help` option:

For **train**:

```shell
$> train --help
Usage: train [OPTIONS]

Options:
  --dataset-path TEXT          Path for the dataset used for training
                               [required]

  --output-path TEXT           Path where checkpoints, metrics and model
                               artifact will be saved

  --batch-size INTEGER         Training batch size
  --images-height INTEGER      Final height of images after resize  [required]
  --images-width INTEGER       Final width of images after resize  [required]
  --epochs INTEGER             Number of training epochs (repeats of dataset)
  --learning-rate FLOAT        Learning rate for gradient optimization
                               [required]

  --debug / --no-debug         Whether or not to use tensorflow in debug mode
  --layer-sizes TEXT           Comma-separeted list of dense layer sizes for
                               the model

  --number-of-classes INTEGER  Number of output classes (the number of neurons
                               in the output layer)

  --summary-step-size INTEGER  Number of steps between each metric report and
                               checkpoint save

  --help                       Show this message and exit.

```

For **validate**:

```shell
$> validate --help
Usage: validate [OPTIONS]

Options:
  --saved-model-path TEXT  Path to the saved model artifact  [required]
  --dataset-path TEXT      Path to the validation dataset  [required]
  --batch-size INTEGER     Validation batch size
  --help                   Show this message and exit.

```

## Dataset

The model expects the data as images in any format recognized by Tensorflow. It should be organized in folders, where
each folder has the name of the corresponding class. You may find a complying dataset in the
[mnist-png repository](https://github.com/IABrasil/mnist-png) from [IABrasil](https://github.com/IABrasil).
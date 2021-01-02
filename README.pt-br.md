# MNIST Code Quality

\* Há uma versão em [Inglês](README.md) aqui.  
\* There is a [English](README.md) in this link.

Este repositório contém uma implementação em Tensorflow de uma solução para o problema de reconhecimento de dígitos do dataset MNIST, porém escrito seguindo os princípios da qualidade de código. O problema em si é bem fácil, e pode ser resolvido com soluções simples. O objetivo aqui é mostrar como organizar o código de uma aplicação para treinamento e validação de um modelo de ML, levando a um resultado facilmente refatorável e organizado por responsabilidades.

## Conteúdo

Este projecto consiste dos seguintes módulos:

* `metrics`: Contém as métricas usadas no treinamento e validação.
* `models`: Contém o modelo, está escrito de forma a fornecer todos os dados necessários ao seu deploy.
* `persistence`: Funções para ler e preprocessar os dados para treinamento e validação.
* `training`: Contém o loop e ponto de entrada para treinamento.
* `validating`: Contém o loop e o ponto de entrada para validação.

## Como executar

Primeiro, crie um `virtualenv` para evitar conflitos com outros projetos e a instalação do sistema, depois, instale usando pip.

```shell
$> git clone https://github.com/andreclaudino/mnist-code-quality.git
$> cd mnist-code-quality/
$> python3 -m virtualenv venv
$> source venv/bin/activate
$> pip3 install -e .
```

Após a instalação, haverá dois comandos, `train` (para treinar um novo modelo) e `validate`, (para validar um modelo já treinado). Os parâmetros necessários para ambos podem ser encontrados com o uso do parâmetro `--help`:
 
Para **train**:

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

para **validate**:

```
$> validate --help
Usage: validate [OPTIONS]

Options:
  --saved-model-path TEXT  Path to the saved model artifact  [required]
  --dataset-path TEXT      Path to validating dataset  [required]
  --batch-size INTEGER     Validating batch size
  --help                   Show this message and exit.

```

## Dataset

O software espera os dados como imagens em qualquer formato reconhecido pelo Tensorflow. ELe deve estar organizado em pastas, o nome de cada pasta corresponde ao nome da classe das imagens dentro dela. Você pode encontrar um dataset adequado a este projeto no [repositório mnist-png](https://github.com/IABrasil/mnist-png) da comunidade [iaBrasil](https://github.com/IABrasil).
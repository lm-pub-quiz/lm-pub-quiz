# Getting started

This library implements a knoweledge probing approach which uses LM's inherent ability to estimate the log-likelihood of any given textual statement.
For more information visit the [LM Pub Quiz website](https://lm-pub-quiz.github.io/).

In this guide, we will first cover how to set up the packaged (as there are multiple options).
For information on how to use the pacakge, see [an example workflow using the API](example.md) or the [CLI-guide](cli.md).

## Recommend Setup

In this short guide, we explain how to setup your environment and run the first command.

If you just want to use the CLI utilities, we recommend using [`pipx`](https://pypa.github.io/pipx/).
If you want to make changes to the package (in order to contribute or customize the experiments),
we recommend cloing it and use either an environment automatically managed by [`hatch`](https://hatch.pypa.io) or a manually managed environment.


### Install in a manually managed environment

#### (Optional) Set up your desired environment

In this example we are using [`conda`](https://docs.conda.io) (and assume you want to install the CPU version of PyTorch; modify accordingly):

```shell
# create an new environment
conda create --name knowledge-probe

# activate the new environment
conda activate knowledge-probe

# install pytorch in conda
conda install pytorch cpuonly -c pytorch
```

#### Install the Package

You can install the package locally or directly from PyPI.

##### From PyPI


```shell
pip install lm-pub-quiz

```

##### From the Source Code

Alternatively, you can clone the repository, then (in your desired environment) run:

```shell
pip install -e lm-pub-quiz # local package (replace lm-pub-quiz with the path to the repository)
```

This allows you to make changes source code which will be reflected directly.


### Install in a *pipx*-managed environment

If you only want to use the command line interface, we recommend to install the package using `pipx`.
With `pipx`, you can install the package directly using `pipx install lm-pub-quiz`.
This will make the commands available on your system, but isolate all dependencies.


### Install in a *hatch*-managed environment

If you want to contribute to *lm-pub-quiz*, we recommend to use [hatch](https://hatch.pypa.io). In this case you need to:

1. Clone the repository,
2. in the directoy run your respective commands in a hatch shell (either `hatch run <command>` or run `hatch shell` and continue your work there).

This allows you to run the test cases by executing 
To run the test cases, run `hatch run test` or `hatch run all:test` (to test on multiple python versions) and to check the formatting and correct typing using `hatch run lint:all`.

## Verify the Installation

To verify the package has been installed correctly, execute:

``` shell
evaluate_model --help
```

You should than see an output similar to this:

``` shell-session
$ evaluate_model --help

usage: evaluate_model [-h] [config_file] <configuration options to overwrite>

Evaluate a given model on a dataset.

positional arguments:
  config_file           Top-level config file to load (optional).

optional arguments:
  -h, --help            show this help message and exit
  --series-skip N       Skip first N trials in the execution of a series.

configuration:
  --model PATH
  --model.name_or_path MODEL.NAME_OR_PATH
  --model.tokenizer MODEL.TOKENIZER
  --model.reduction MODEL.REDUCTION
  --model.pll_metric MODEL.PLL_METRIC
  --dataset PATH
  --dataset.path DATASET.PATH
  --output_base_path OUTPUT_BASE_PATH

```


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

##### From PyPI (recommended for most usage)


```shell
pip install lm-pub-quiz

```


### Install in a *hatch*-managed environment (Recommended for Development)

If you want to contribute to *lm-pub-quiz*, we recommend to use [hatch](https://hatch.pypa.io). In this case you need to:

1. Clone the repository,
2. in the directoy run your respective commands in a hatch shell (either `hatch run <command>` or run `hatch shell` and continue your work there).

This allows you to run the test cases by executing 
To run the test cases, run `hatch run test` or `hatch run all:test` (to test on multiple python versions) and to check the formatting and correct typing using `hatch run lint:all`.



##### From the Source Code

Alternatively, you can clone the repository, then (in your desired environment) run:

```shell
pip install -e lm-pub-quiz # local package (replace lm-pub-quiz with the path to the repository)
```

This allows you to make changes source code which will be reflected directly.



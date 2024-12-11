If you want to make changes to the package (in order to contribute or customize the experiments),
we recommend cloning it and using the environments automatically created by [`hatch`](https://hatch.pypa.io).


### Using *hatch*-managed environment (Recommended for Development)

If you want to contribute to *lm-pub-quiz*, we recommend to use [hatch](https://hatch.pypa.io). In this case you need to:

1. [Install hatch](https://hatch.pypa.io/latest/install/#pipx) (if you haven't already), and
2. clone the repository: `git clone git@github.com:lm-pub-quiz/lm-pub-quiz.git`


In the cloned directory, you can now run the relevant commands in a hatch shell.
You can either run a command with `hatch run <command>` or run `hatch shell` and continue to work within the activated environment.

This allows you to run the test cases (`hatch run test`), format the code (`hatch run lint:fmt`), and check for typing inconsistencies (`hatch run lint:all`). 

By specifying the environment before the command (as in `lint:`), commands can be run in specific environment that hatch manages for you. By running `hatch run all:test` the library can be tested on multiple python versions.

Use `hatch run serve-docs` to start a local web server serving the current state of this documentation.

For more information on the usage of hatch, we refer to the [documentation of hatch](https://hatch.pypa.io/latest/).

### Without *hatch*

Alternatively, you can clone the repository, then (in your desired environment) and run:

```shell
# replace lm-pub-quiz with the path to the repository
pip install -e lm-pub-quiz
```

This allows you to make changes source code which will be reflected directly within your manually managed environment.



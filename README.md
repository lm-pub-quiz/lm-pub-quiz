# LM Pub Quiz
*Evaluate language models using multiple choice items*


[![Build status](https://img.shields.io/github/actions/workflow/status/lm-pub-quiz/lm-pub-quiz/test.yml?logo=github&label=Tests)](https://github.com/lm-pub-quiz/lm-pub-quiz/actions)
[![PyPI - Version](https://img.shields.io/pypi/v/lm-pub-quiz.svg?logo=pypi&label=Version&logoColor=gold)](https://pypi.org/project/lm-pub-quiz/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lm-pub-quiz?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/lm-pub-quiz/)
[![License](https://img.shields.io/github/license/lm-pub-quiz/lm-pub-quiz?logo=pypi&logoColor=gold)](https://github.com/lm-pub-quiz/lm-pub-quiz/blob/main/LICENSE)

---

This library implements a knoweledge probing approach which uses LM's inherent ability to estimate the log-likelihood of any given textual statement.
For more information visit the [LM Pub Quiz website](https://lm-pub-quiz.github.io/).

### See also
- [Website](https://lm-pub-quiz.github.io/)
- [Documentation](https://lm-pub-quiz.github.io/lm-pub-quiz)
- [BEAR Dataset](https://github.com/lm-pub-quiz/BEAR)
- [Paper](https://arxiv.org/abs/2404.04113)


## Getting started

This short guide should get you started. For more detailed information visit the [documentation](https://lm-pub-quiz.github.io/lm-pub-quiz). 

### Installing the Package

You can install the package via *pip*:

```shell
pip install lm-pub-quiz
```

For alternatives methods of installing the package, visit the [documentation](https://lm-pub-quiz.github.io/lm-pub-quiz).


### Example Usage

```python
from lm_pub_quiz import Dataset, Evaluator

# Load the dataset
dataset = Dataset.from_name("BEAR")

# Load the model
evaluator = Evaluator.from_model(
    "gpt2",
    model_type="CLM",
)

# Run the evaluation and save the
results = evaluator.evaluate_dataset(
    dataset,
    save_path="gpt2_results",
    batch_size=32,
)

# If the results are analyzed in a different session, they can be loaded from the file system
# results = DatasetResults.from_path("gpt2_results")

print("=== Overall score ===")
print(results.get_metrics("accuracy"))
```


## Contributing
We welcome any questions, comments, or even PRs to this project to improve the package.

We use [hatch](https://hatch.pypa.io) to manage this project. For the most comfortable development experience, please first install hatch using [`pip`](https://hatch.pypa.io/latest/install/#pipx) or [`pipx`](https://hatch.pypa.io/latest/install/#pipx).

Then, to propose a change to the library,

- test your code locally using `hatch run all:test`
- format the code according to our formatting guidelines using `hatch run lint:fmt`,
- check type- and style-consistency using `hatch run lint:all`, and
- finally create a pull request describing the changes you propose.

For work on the documentation, use `hatch run serve-docs` to run a local documentation server.

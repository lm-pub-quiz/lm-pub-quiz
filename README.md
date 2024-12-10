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

or clone the repository and install the package using the `-e` flag to make changes to the source code:

```shell
pip install -e lm-pub-quiz  # Modify the path to the repository if necessary
```

For alternatives methods of installing the package, visit the [documentation](https://lm-pub-quiz.github.com/lm-pub-quiz).

### Example Usage

```python
from lm_pub_quiz import Dataset, Evaluator

<<<<<<< HEAD
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
    template_index=0,
    save_path="gpt2_results",
    batch_size=32,
)

# results = DatasetResults.from_path("gpt2_results")

print("=== Overall score ===")
print(results.get_metrics("accuracy"))
```


## Contributing
We welcome any questions, comments, or event PRs to this project to improve the package.

We use [hatch](https://hatch.pypa.io) to manage this project.
To run the test cases, run `hatch run test` or `hatch run all:test` (to test on multiple python versions).
In order to check the formatting and correct typing, run `hatch run lint:all`.

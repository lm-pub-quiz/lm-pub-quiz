# Command Line Interface


There are three commands introduced by this package:

| Command | Description |
| :------ | :---------- |
| [`score_sentence`](#score_sentence) | A thin wrapper around [`minicons` package](https://github.com/kanishkamisra/minicons) -- scores a given sentence. |
| [`rank_answers`](#rank_answers) | Rank answers for a given template. |
| [`evaluate_dataset`](#evaluate_dataset) | Run a complete evaluation on a given dataset. |


Run a command with `--help` to get a description of the command and all its configuration options.


## Configuration

It is possible to load a configuration file specifying the complete or parital configuration and overwrite configurations using the command line options (using two leading dashes, e.g. `--device DEVICE_TO_USE`).

All commands share the same base arguments to specify the model and retrieval details:


| Option   | Description | 
| :------- | :---------- |
| `model.name_or_path` | The model to use (can be a huggingface name of a local path; mandatory).
| `model.tokenizer` | The name of the tokenizer to use (defaults to one matching the model). |
| `model.reduction` | Type of score reduction (can be `mean` or `sum`; defaults to the latter). |
| `model.pll_metric` | The type of PLL metric to use (only use when using MLM-type model; defaults to `within_word_l2r`). |
| `model.lm_type` | If the model type cannot be inferred, pass `MLM` or `CLM` depending on the type of your model. |
| `model` | Specify a path to load these (the `model.`) configurations from (the `model.` prefix must then be omitted). |
| `device` | Specify which device to use (default to using the CPU). |

Each command has a set of additional options to specify what to score/evaluate/rank.


### Configuration Files

Configuration files can be in either of the following formats (the extension must specify the format):

- JSON (`.jons`)
- YAML (`.yaml` or `.yml`)
- TOML (`.toml`)

## `evaluate_dataset`

| Option       | Description | 
| :----------- | :---------- |
| `dataset_path` | Path to the dataset which is used for the evaluation. |
| `batch_size` | How many sentences score per batch. |
| `output_base_path` | Base path for storing the results. |
| `debug` | Set `debug` to true (or use `--debug`) to only evaluate two random instances per relation. |

## `rank_answers`

| Option       | Description | 
| :----------- | :---------- |
| `template` | The template to use. Must be of the form "Some sentence with a [Y] marker". "[Y]" wil be replaced by each answer. |
| `answers` | A list of strings separated by ','. |


## `score_sentence`

| Option       | Description | 
| :----------- | :---------- |
| `sentence` | The sentence to score. |


### Examples

``` shell-session
$ score_sentence --model.name "distilbert-base-cased" --sentence "The traveler lost the souvenir." --model.reduction "none"
Rank    Score   
----------------
The         3.26
travel      8.34
##er        3.81
lost        8.16
the         2.18
so          8.66
##uve       3.54
##nir      -0.00
.           1.51

$ score_sentence --model.name "distilbert-base-cased" --sentence "The traveler lost the souvenir." --model.reduction "none" --model.pll_metric "original"
Rank    Score   
----------------
The         3.26
travel      3.46
##er        3.81
lost        8.16
the         2.18
so          0.03
##uve       0.00
##nir      -0.00
.           1.51
```



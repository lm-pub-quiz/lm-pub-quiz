# Example Workflow
In this short guide we will go through a brief workflow example, in which we run the BEAR probe on the gpt2 model and look at some results using the python api.

## Run BEAR probe on a given model
First we run the BEAR probe on the given model and save its results to our file system.

```python
from lm_pub_quiz import Dataset, Evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the BEAR dataset from its specific location
dataset = Dataset.from_path("<BEAR data path, e.g. ./transformer-knowledge-probe/data/BEAR>")

# Load the gpt2 model
model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")


# Run the BEAR evaluator and save the results
result_save_path = "<BEAR results save path>"  # doctest: +SKIP

# In case the tokenizer cannot be loaded from the model directory, it may be loaded explicitly and passed to the Evaluator.from_model method via the 'tokenizer=' keyword
evaluator = Evaluator.from_model(model, model_type="CLM", device="cuda")
evaluator.evaluate_dataset(dataset, save_path=result_save_path, batch_size=32)
```


## Inspect BEAR Probe results
We can load the BEAR probe results as follows:

```python
from lm_pub_quiz import DatasetResults
bear_results = DatasetResults.from_path(result_save_path)
```

### Aggregate Results
The DatasetResults object allows us to retrieve some aggregate results. Here we are loading the accuracy and the precision_at_k metrics:
```python
metrics = bear_results.get_metrics(["accuracy", "num_instances"])
```
The method returns a pandas dataframe that holds the specified metrics for each relation (P6 to P7959) in the BEAR dataset (here showing the first five entries):
```
     accuracy  num_instances relation_type
P6   0.183333             60          None
P19  0.206667            150          None
P20  0.160000            150          None
P26  0.050000             60          None
P27  0.406667            150          None
```

To aggregate these accuracy scores over all relations we weigh them by the number of instances within each relation. Otherwise, greater accuracies more easily achieved on a small relation would inflate the overall accuracy.

```python
import numpy as np
weighted_accuracy = np.average(metrics.accuracy, weights=metrics.num_instances)
```
For the *gpt2* model we thus get a `weighted_accuracy` of `0.1495`. Note that this overall accuracy score is based only on the first template of each relation, which is the template considered by default by the `Evaluator`.

### Individual Results

The DatasetResults object holds `RelationResult` objects for each relation in the probe that can be accessed using the relation codes in a key-like manner. If we want to take a more detailed look at the results for individual relations we may look at the instance tables these RelationResults hold:

```python
relation_instance_table = bear_results["P36"].instance_table
print(relation_instance_table.head())
```

```
      sub_id                  sub_label  answer_idx                                         pll_scores  obj_id      obj_label
3      Q1356                West Bengal           0  [-28.071779251, -35.064821243299996, -32.31778...   Q1348        Kolkata
11     Q1028                    Morocco           1  [-33.614648819, -26.9230899811, -32.1363086701...   Q3551          Rabat
15  Q3177715         Pagaruyung Kingdom           2  [-65.55403518690001, -67.46153640760001, -66.3...   Q3492        Sumatra
18   Q483599  Southern Federal District           3  [-46.7988452912, -49.6077213287, -49.030160904...    Q908  Rostov-on-Don
20    Q43684                      Henan           4  [-36.29014015210001, -37.7681064606, -41.59478...  Q30340      Zhengzhou
```
Here we see the instance table for the relation `P36`. Each row of this instance table holds the results for a specific instance of this relation, i.e. the log-likelihood scores of instantiations of a template of the relation with the subject of this row and the objects in the relation answer space. The columns can be interpreted as follows:

- `sub_id`: wikidata code of the subject instance of this row
- `sub_label`: label of that subject instance
- `sub_aliases`: alternative labels for that subject instance
- `answer_idx`: id in the pll_scores list for the score of the true answer for this instance
- `pll_scores`: (pseudo) log-likelihood scores for all objects in the answer space.
- `obj_id`: wikidata code for the true object in the answer space
- `obj_label`: label for the true object in the answer space

Note that the `pll_scores` are ordered corresponding to the orders of the objects in this relations answer space (`bear_results["P36"].answer_space`).

We will lastly be looking at two examples of what we can do with this data: (1) Collect the specific instances the model got right for each relation. (2) Estimate the prior for each object in the answer space for each relation.

### Correct Instances
To gain more insight into the individual strengths and weaknesses of the model under investigation, we may want to inspect which specific instances of a relation the model got right and where it was wrong. Given the instance table this information is easy to retrieve. We only need to compare the index of the greatest pll_score to the `answer_idx` to determine whether for a given subject the correct object was scored as most likely:

```python
relation_instance_table["correctly_predicted"] = relation_instance_table.apply(lambda row: row.answer_idx == np.argmax(row.pll_scores), axis=1)
```

```
      sub_id                  sub_label  answer_idx                                         pll_scores  obj_id      obj_label  correctly_predicted
3      Q1356                West Bengal           0  [-28.071779251, -35.064821243299996, -32.31778...   Q1348        Kolkata                False
11     Q1028                    Morocco           1  [-33.614648819, -26.9230899811, -32.1363086701...   Q3551          Rabat                 True
15  Q3177715         Pagaruyung Kingdom           2  [-65.55403518690001, -67.46153640760001, -66.3...   Q3492        Sumatra                False
18   Q483599  Southern Federal District           3  [-46.7988452912, -49.6077213287, -49.030160904...    Q908  Rostov-on-Don                False
20    Q43684                      Henan           4  [-36.29014015210001, -37.7681064606, -41.59478...  Q30340      Zhengzhou                False
```

### Answer Space Priors
Another question we may ask ourselves is to what extent a models log-likelihood scores for individual instantiations of a relation depend on what the model has learned about the connection between the specific subject and object or to what extent these scores are determined by a general bias the model possess towards certain objects in the answer space.

To address this we may want to estimate the priors for all objects in the answer space.

For a causal language model such as *gpt2* the `pll_scores` are identical to the log-likelihood of the sentences derived by instantiating the template with the given subjects and objects.
Taking the relation *P30* as an example, with the template `[X] is located in [Y].`, the subject `Nile` and the object `Africa`, the `pll_score`for this pairing is the log of the probability assigned by the evaluated language model to the sentence `Nile is located in Africa`.

Calculating the softmax over the `pll_scores` for a given subject of a relation gives us the conditional probabilities of the instantiated sentences of the relation conditioned on the fact that one of the instantiations is correct.

Averaging these probability distributions over all subjects in the subject space of the relation estimates the priors for the objects in the answer space.

```python
import torch

relation_code = "P30"

softmax = torch.nn.Softmax(dim=0)
relation_instance_table = bear_results[relation_code].instance_table
relation_instance_table["pll_softmax"] = relation_instance_table.pll_scores.apply(lambda x: softmax(torch.tensor(x)))
relation_priors = pd.Series(
    torch.mean(torch.stack(list(relation_instance_table.pll_softmax)), dim=0),
    index=bear_results[relation_code].answer_space.values
)
```

For the relation *P30* this results in the following priors.

```
Africa           0.162270
Antarctica       0.111288
Asia             0.120442
Europe           0.220671
North America    0.218366
South America    0.166962
dtype: float64
```

We can see that *gpt2* is biased towards answering "North America" and "Europa" when assessing the entities in the subject space of relation *P30* with the template `[X] is located in [Y].` (though the objects are balanced).


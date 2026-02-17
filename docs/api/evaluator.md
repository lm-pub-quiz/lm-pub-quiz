# Evaluator

The [`Evaluator` class][lm_pub_quiz.Evaluator] is the central processor running the evaluation of a model on a dataset.
It uses a [`ModelInterface`][lm_pub_quiz.ModelInterface] to score options within a set of answers.

To create an `Evaluator` for a given model, the `Evaluator.from_model` method can be used. 
The appropriate `ModelInterface` class is then chosen automatically.


```python
evaluator = Evaluator.from_model("gpt", model_type="CLM")
```


::: lm_pub_quiz.Evaluator




# Model Interfaces


A `ModelInterface` implements the interaction with the model: Taking a set of statements, it needs to score each of the options.
Depending on the type of the model (causal vs. masked language models), different `ModelInterface`s are required.



## Base Interface 

::: lm_pub_quiz.ModelInterface

The `PLLModelInterfaceMixin` can be used to join the statement sets into one iterable, requiring the interface to just score individual texts.

::: lm_pub_quiz.model_interface.PLLModelInterfaceMixin



## Sentence-Loglikelihood-based Interfaces

The following interfaces implement (pseudo) loglikelihood scoring for the text options.

::: lm_pub_quiz.model_interface.hf.CLMInterface


::: lm_pub_quiz.model_interface.hf.MLMInterface


## Other Interfaces


::: lm_pub_quiz.model_interface.hf.tyq.TyQModelInterface

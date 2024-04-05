import warnings
from typing import Any, Dict, List, Mapping, Optional, cast

import numpy as np

from lm_pub_quiz.metrics.base import RelationMetric
from lm_pub_quiz.util import sort_scores


class BasicMetric(RelationMetric):
    metric_name = "num_instances"

    def reset(self):
        self.num_instances = 0

    def add_instance(self, row: Mapping):  # noqa: ARG002
        self.num_instances += 1

    def compute(self):
        return {"num_instances": self.num_instances}


class PrecisionAtK(BasicMetric):
    metric_name = "precision_at_k"

    def reset(self):
        super().reset()
        self.num_correct_at_k: Optional[np.ndarray] = None

    def add_instance(self, row: Mapping):
        super().add_instance(row)
        ranked_indices, _ = zip(*sort_scores(cast(List[float], row["pll_scores"])))

        if self.num_correct_at_k is None:
            # Initialize the array
            self.num_correct_at_k = np.zeros(len(ranked_indices), dtype=int)

        elif len(ranked_indices) > self.num_correct_at_k.shape[0]:
            # extend the array
            warnings.warn("Computing precision at k with varying numbers of answers.", stacklevel=1)

            old_values = self.num_correct_at_k
            self.num_correct_at_k = np.zeros(len(ranked_indices), dtype=int)
            self.num_correct_at_k[: old_values.shape[0]] = old_values

        for r, idx in enumerate(ranked_indices):
            # Increase the counts for each rank above the rank of the correct answer
            if idx == row["answer_idx"]:
                self.num_correct_at_k[r:] += 1

    def compute(self):
        if self.num_correct_at_k is None:
            return super().compute()
        else:
            return {**super().compute(), "precision_at_k": (self.num_correct_at_k / self.num_instances).tolist()}


class Accuracy(BasicMetric):
    metric_name = "accuracy"

    def reset(self):
        super().reset()
        self.num_correct: int = 0

    def add_instance(self, row: Mapping):
        super().add_instance(row)
        if row["answer_idx"] == row["pll_scores"].index(max(row["pll_scores"])):
            self.num_correct += 1

    def compute(self) -> Dict[str, Any]:
        return {**super().compute(), "accuracy": self.num_correct / self.num_instances}

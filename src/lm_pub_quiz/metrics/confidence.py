from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import brier_score_loss

from lm_pub_quiz.metrics.basic import BasicMetric


class SoftmaxBase(BasicMetric):
    """Base class computes the softmax probability per row (instance)."""

    metric_name = "avg_softmax"

    def reset(self):
        super().reset()
        self.softmax: Optional[pd.Series] = None
        self.correct_mask: List = []

    @staticmethod
    def _get_softmax_probability(pll_scores: List[float]) -> np.array:
        exp_scores = np.exp(pll_scores)
        softmax_scores = exp_scores / np.sum(exp_scores)
        return softmax_scores

    @staticmethod
    def _stack_vector(base_vector: np.ndarray, new_vector: np.ndarray) -> np.ndarray:
        if (base_vector is None) and (new_vector.ndim == 1):
            base_vector = new_vector.reshape(1, -1)
        elif (base_vector is None) and (new_vector.ndim > 1):
            base_vector = new_vector
        else:
            base_vector = np.vstack((base_vector, new_vector))

        return base_vector

    def add_instance(self, row: Mapping):
        super().add_instance(row)
        softmax_scores = self._get_softmax_probability(row["pll_scores"])

        correct = row["answer_idx"] == np.argmax(row["pll_scores"])

        self.correct_mask.append(correct)
        self.softmax = self._stack_vector(self.softmax, softmax_scores)

    def compute(self) -> Dict[str, Any]:
        super().compute()

        if self.softmax is None:
            msg = "Unable to compute probability distributions."
            raise ValueError(msg)

        self.correct_mask = np.array(self.correct_mask)
        self.softmax_sorted = np.sort(self.softmax, axis=1)[:, ::-1]  # get ranks in desceding order
        return {
            "avg_softmax": {
                "all": self.softmax.mean(axis=0),
                "correct": self.softmax[self.correct_mask].mean(axis=0) if sum(self.correct_mask) else None,  # type: ignore
                "incorrect": self.softmax[~self.correct_mask].mean(axis=0) if sum(~self.correct_mask) else None,  # type: ignore
            },
        }


class ConfidenceMargin(SoftmaxBase):
    """Margin of Confidence Metric: Difference between the highest and
    the second-highest predicted probability scores.

    A larger margin indicates that the model is more confident in its decision,
    distinguishing clearly between the most likely and the second most likely answers.
    """

    metric_name = "margin_of_confidence"

    def compute(self) -> Dict[str, Any]:
        super().compute()
        confidence_margin = self.softmax_sorted[:, 0] - self.softmax_sorted[:, 1]
        return {
            "margin_of_confidence": {
                "all": confidence_margin.mean(axis=0),
                "correct": confidence_margin[self.correct_mask].mean(axis=0) if sum(self.correct_mask) else None,  # type: ignore
                "incorrect": confidence_margin[~self.correct_mask].mean(axis=0) if sum(~self.correct_mask) else None,  # type: ignore
            },
        }


class ConfidenceScore(SoftmaxBase):
    """Confidence Score: This is the maximum probability value in the softmax output.

    This corresponds to the first (0-indexed) value in the sorted softmax values.

    It directly indicates the model's confidence in its most likely prediction.
    Higher confidence scores indicate higher confidence, whereas scores closer to 1/N suggest lower confidence.
    This is Equation 2.1.4
    """

    metric_name = "confidence_score"

    def compute(self) -> Dict[str, Any]:
        super().compute()
        correct = self.softmax_sorted[:, 0][self.correct_mask].mean(axis=0) if sum(self.correct_mask) else None  # type: ignore
        incorrect = self.softmax_sorted[:, 0][~self.correct_mask].mean(axis=0) if sum(~self.correct_mask) else None  # type: ignore
        return {
            "confidence_score": {
                "all": self.softmax_sorted[:, 0].mean(axis=0),
                "correct": correct,
                "incorrect": incorrect,
            },
        }


class UncertaintyScore(SoftmaxBase):
    """UncertaintyScore: This is a normalized entropy of the predicted prob. distribution.

    This is Equation 2.1.1
    """

    metric_name = "uncertainty_score"

    def reset(self):
        super().reset()
        self.correct_answer: List = []

    def add_instance(self, row: Mapping):
        super().add_instance(row)
        self.correct_answer.append(row["answer_idx"])

    def compute(self) -> Dict[str, Any]:
        super().compute()

        self.correct_answer = np.array(self.correct_answer)

        num_answers = len(np.unique(self.correct_answer))
        entropy_uniform = np.emath.logn(2, num_answers)

        uncertainty = np.apply_along_axis(lambda row: entropy(row, base=2) / entropy_uniform, 1, self.softmax_sorted)

        return {
            "uncertainty_score": {
                "all": uncertainty.mean(axis=0),
                "correct": uncertainty[self.correct_mask].mean(axis=0) if sum(self.correct_mask) else None,  # type: ignore
                "incorrect": uncertainty[~self.correct_mask].mean(axis=0) if sum(~self.correct_mask) else None,  # type: ignore
            },
        }


class BrierScore(UncertaintyScore):
    """BrierScore is calculated as the mean squared difference between the predicted probability
    assigned to the possible outcomes and the actual outcome.

    The Brier Score ranges from 0 to 1, where a lower Brier Score indicates a better model.
    The model with the lower Brier Score is considered to have better predictive accuracy.
    """

    metric_name = "brier_score"

    def compute(self) -> Dict[str, Any]:
        super().compute()
        y_true, y_pred_probs = self.correct_answer, self.softmax

        y_true_one_hot = np.zeros_like(y_pred_probs)
        y_true_one_hot[np.arange(len(y_true)), y_true] = 1

        brier_scores = [
            brier_score_loss(y_true_one_hot[:, i], y_pred_probs[:, i]) for i in range(y_pred_probs.shape[1])  # type: ignore
        ]

        return {"brier_score": np.mean(brier_scores)}

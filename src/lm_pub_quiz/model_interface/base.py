from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Optional, Union

from lm_pub_quiz.types import (
    ItemScores,
    ItemTokenScoresAndRoles,
    StatementScore,
    TextRoles,
    TokenScoresAndRoles,
)
from lm_pub_quiz.util import ReversibleChain


class ModelInterface(ABC):
    """Shared interface for methods that process each answer separately (PLL scoring) or sets of answers (like TYQ)."""

    model_name: str

    @classmethod
    @abstractmethod
    def from_model(cls, model: Any, **kw) -> "ModelInterface":
        """Create a `ModelInterface` from the given parameters.

        Load the model, tokenizer, etc. and instantiate a `ModelInterface`."""
        pass

    @abstractmethod
    def score_statement_options(
        self,
        statement_options: Iterable[Sequence[str]],
        *,
        text_roles: Optional[Iterable[Sequence[TextRoles]]] = None,
        **kw,
    ) -> Union[Iterable[ItemTokenScoresAndRoles], Iterable[ItemScores]]:
        """Score sets of text options.

        The `ModelInterface` itself is resonsible for batching the requests.
        """
        pass

    def get_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
        }


class PLLModelInterfaceMixin:
    """Interface for methods that assign a score per statement."""

    default_reduction: str = "sum"

    @abstractmethod
    def score_statements(
        self,
        statements: Iterable[str],
        *,
        text_roles: Optional[Iterable[TextRoles]] = None,
        **kw,
    ) -> Union[Iterable[StatementScore], Iterable[TokenScoresAndRoles]]:
        """Score individual texts (independent of the other options) using the Casual/Masked Language Model.

        Parameters:
            statements: The statements to score.
            text_roles: Which parts of the statement are the answer, template, and subject.

        Returns:
            Scores (or scores and roles) per statement
        """
        pass

    def score_statement_options(
        self,
        statement_options: Iterable[Sequence[str]],
        *,
        text_roles: Optional[Iterable[Sequence[TextRoles]]] = None,
        **kw,
    ) -> Union[Iterable[ItemScores], Iterable[ItemTokenScoresAndRoles]]:
        """Join the sets of statements, process each statement, and order the scores according to the inputs."""

        if text_roles is None:
            chained = ReversibleChain({"statements": statement_options})
        else:
            chained = ReversibleChain({"statements": statement_options, "text_roles": text_roles})

        return chained.reverse(self.score_statements(**chained, **kw))

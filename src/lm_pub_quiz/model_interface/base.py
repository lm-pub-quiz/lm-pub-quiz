from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
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
        pass

    @abstractmethod
    def score_statement_options(
        self,
        *,
        statements: Iterable[Sequence[str]],
        roles: Optional[Iterable[Sequence[TextRoles]]] = None,
        **kw,
    ) -> Iterator[Union[ItemTokenScoresAndRoles, ItemScores]]:
        pass

    def get_result_metadata(self) -> dict[str, Any]:
        return {}


class PLLModelInterfaceMixin:
    """Interface for methods that assign a score per statement."""

    default_reduction: str = "sum"

    @abstractmethod
    def score_statements(
        self,
        *,
        statements: Iterable[str],
        roles: Optional[Iterable[TextRoles]] = None,
        **kw,
    ) -> Union[Iterable[StatementScore], Iterable[TokenScoresAndRoles]]:
        pass

    def score_statement_options(
        self,
        *,
        statements: Iterable[Sequence[str]],
        roles: Optional[Iterable[Sequence[TextRoles]]] = None,
        **kw,
    ) -> Union[Iterable[ItemScores], Iterable[ItemTokenScoresAndRoles]]:
        """Join the sets of statements, process each statement, and order the scores according to the inputs."""

        if roles is None:
            chained = ReversibleChain({"statements": statements})
        else:
            chained = ReversibleChain({"statements": statements, "roles": roles})

        scores = self.score_statements(**chained, **kw)
        return chained.reverse(scores)

"""MultipleChoiceEvaluator"""

import logging
import warnings
from collections.abc import Iterable, Sequence
from typing import (
    Callable,
    Optional,
    Union,
    overload,
)

from tqdm.auto import tqdm
from transformers import (
    BatchEncoding,
)

from lm_pub_quiz.types import (
    ScoringMask,
    TextRoles,
    TokenRoles,
)

tqdm.pandas()


log = logging.getLogger(__name__)


def derive_token_roles(
    *,
    batch: BatchEncoding,
    text_roles: Sequence[TextRoles],
    scoring_masks: Optional[Iterable[ScoringMask]],
    output_indices: bool,
) -> Sequence[TokenRoles]:
    warnings.warn(
        "We recommennd using the more explicit functions `derive_token_roles_internal` and `remap_token_roles`.",
        stacklevel=2,
    )

    token_roles = derive_token_roles_internal(
        batch=batch,
        text_roles=text_roles,
    )

    if not output_indices:
        return token_roles
    elif scoring_masks is None:
        msg = "Cannot remap token roles if the scoring masks are not set."
        raise ValueError(msg)
    else:
        return [
            remap_token_roles(token_roles_internal=roles, scoring_mask=mask)
            for roles, mask in zip(token_roles, scoring_masks)
        ]


@overload
def derive_token_roles_internal(
    *,
    batch: Iterable[BatchEncoding],
    text_roles: Iterable[Sequence[TextRoles]],
) -> Iterable[Sequence[TokenRoles]]: ...


@overload
def derive_token_roles_internal(
    *,
    batch: BatchEncoding,
    text_roles: Sequence[TextRoles],
) -> Sequence[TokenRoles]: ...


def derive_token_roles_internal(
    *,
    batch: Union[BatchEncoding, Iterable[BatchEncoding]],
    text_roles: Union[list[TextRoles], Iterable[list[TextRoles]]],
) -> Union[Sequence[TokenRoles], Iterable[TokenRoles]]:
    """Derive which tokens belong to the subject, answer, and template.

    If the scoring mask is given, the token indices refer to the resulting scores.
    """
    if isinstance(batch, BatchEncoding):
        assert isinstance(text_roles, list)

    else:
        for b, tr in zip(batch, text_roles):
            yield derive_token_roles_internal(batch=b, text_roles=tr)
        return

    if not len(batch) == len(text_roles):
        msg = f"Number of statements in batch ({len(batch)}) and text roles ({len(text_roles)}) does not match."
        raise ValueError(msg)

    token_roles: list[TokenRoles] = []

    non_template_tokens: set[int]
    roles: dict[str, list[int]]

    for statement_index, spans in enumerate(text_roles):
        non_template_tokens = set()

        # For the statement, determine which tokens belong to each role
        roles = {k: [] for k in spans.keys()}

        for k, v in spans.items():
            for start, end in v:
                # go through the span until we find the first token
                first_affected_token: Optional[int] = next(
                    (t for i in range(start, end) if (t := batch.char_to_token(statement_index, i)) is not None),
                    None,
                )

                # do the same, just in reversed order
                last_affected_token: Optional[int] = next(
                    (
                        t
                        for i in reversed(range(start, end))
                        if (t := batch.char_to_token(statement_index, i)) is not None
                    ),
                    None,
                )

                if first_affected_token is None or last_affected_token is None:
                    # There was no token within the span... continue
                    continue

                tokens = range(first_affected_token, last_affected_token + 1)

                roles[k].extend(tokens)

                if k != "template":
                    non_template_tokens.update(tokens)

        roles["template"] = [i for i in range(batch.length[statement_index]) if i not in non_template_tokens]
        token_roles.append(roles)

    return token_roles


def remap_token_roles(*, token_roles_internal: TokenRoles, scoring_mask: ScoringMask) -> TokenRoles:
    """Remap the token indices.

    The original token indices in `token_roles_internal` herby refer to the tokens within the input sequence.
    The output token indices refer to the list of tokens that are actually scored (for retrieval of token
    log-likihodds)."""

    i = 0
    remapped: list[int] = []
    for m in scoring_mask:
        if m:
            remapped.append(i)
            i += 1
        else:
            remapped.append(-1)

    token_indices = {
        k: [remapped[i] for i in v if scoring_mask[i] if remapped[i] > 0] for k, v in token_roles_internal.items()
    }

    return token_indices


def get_reduction_function(reduction: str) -> Callable:
    if reduction == "sum":
        return sum
    elif reduction == "mean":
        return lambda x: sum(x) / len(x)
    elif reduction is None:
        return lambda x: x
    else:
        msg = f"Invalid reduction option '{reduction}'. \
            Choose either 'sum', 'mean' or None (for each token)."
        raise ValueError(msg)

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from transformers import BatchEncoding, PreTrainedTokenizer

Span = Tuple[int, int]


log = logging.getLogger(__name__)


class Templater:
    def __init__(self, subject_placeholder="[X]", answer_placeholder="[Y]"):
        self._subject_placeholder = subject_placeholder
        self._answer_placeholder = answer_placeholder

        # build the search pattern for the placeholders
        self._search_pattern = re.compile(
            "(" + re.escape(subject_placeholder) + "|" + re.escape(answer_placeholder) + ")"
        )

    @staticmethod
    def capitalize(text: str) -> str:
        if len(text) > 0:
            return text[0].upper() + text[1:]
        else:
            return text

    def replace_placeholders(
        self,
        *,
        template: str,
        subject: Optional[str],
        answer: Optional[str],
        capitalize: bool = True,
    ) -> Tuple[str, Dict[str, List[Span]]]:
        """Replace all placeholders in the template with the respective values.

        Returns the final string as well as the spans of the respective elements in the final string.
        """
        # initialize the string
        text: str = ""
        # initialize dictionary with final spans
        spans: Dict[str, List[Tuple[int, int]]] = {
            "subject": [],
            "answer": [],
        }

        last_match_end: Optional[int] = None

        # count the offset introduced by replacements
        offset = 0

        for match in self._search_pattern.finditer(template):
            key = match.group()
            if key == self._subject_placeholder:
                value = subject
                span_list = spans["subject"]
            elif key == self._answer_placeholder:
                value = answer
                span_list = spans["answer"]
            else:
                msg = f"Unkown placeholder found: {key}"
                raise RuntimeError(msg)

            if value is not None:
                start, end = match.span()
                text += template[last_match_end : match.start()] + value
                last_match_end = match.end()

                span_list.append((start + offset, start + offset + len(value)))

                offset += len(value) - (end - start)

        # append the remainder of the template.
        text += template[last_match_end:]

        if capitalize:
            text = self.capitalize(text)

        return text, spans

    def tokenize_with_span_dict(
        self,
        *,
        tokenizer: PreTrainedTokenizer,
        text: str,
        spans: Dict[str, List[Span]],
        include_template_indices: bool = True,
        include_special_tokens: bool = True,
    ) -> Tuple[BatchEncoding, Dict[str, List[int]]]:

        encoded = tokenizer(text, return_length=True, add_special_tokens=include_special_tokens)

        non_template_tokens: Set[int] = set()

        if include_template_indices and "template" not in spans:
            spans = {"template": [(0, len(text))], **spans}

        token_indices: Dict[str, List[int]] = {k: [] for k in spans.keys()}

        for k, v in spans.items():
            for start, end in v:
                # go through the span until we find the first token
                first_affected_token: Optional[int] = next(
                    (t for i in range(start, end) if (t := encoded.char_to_token(i)) is not None), None
                )

                # do the same, just in reversed order
                last_affected_token: Optional[int] = next(
                    (t for i in reversed(range(start, end)) if (t := encoded.char_to_token(i)) is not None), None
                )

                if first_affected_token is None or last_affected_token is None:
                    # There was no token within the span... continue
                    continue

                tokens = range(first_affected_token, last_affected_token + 1)

                token_indices[k].extend(tokens)

                if k != "template":
                    non_template_tokens.update(tokens)

        if include_template_indices:
            token_indices["template"] = [i for i in token_indices["template"] if i not in non_template_tokens]

        return encoded, token_indices

    def fill(
        self,
        *,
        template: str,
        subject: Optional[str],
        answer: Optional[str],
        return_spans: bool = False,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        include_template_indices: bool = True,
        capitalize: bool = True,
        **kw,
    ):
        text, spans = self.replace_placeholders(
            template=template, subject=subject, answer=answer, capitalize=capitalize
        )

        if tokenizer is None:
            if not return_spans:
                return text
            else:
                return text, spans
        else:
            _, token_indices = self.tokenize_with_span_dict(
                tokenizer=tokenizer, text=text, spans=spans, include_template_indices=include_template_indices, **kw
            )

            if return_spans:
                return text, spans, token_indices
            else:
                return text, token_indices

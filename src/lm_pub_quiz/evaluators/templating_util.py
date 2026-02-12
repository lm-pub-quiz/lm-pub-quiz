import logging
import re
from typing import Optional

from lm_pub_quiz.evaluators.util import BaseMixin
from lm_pub_quiz.types import Span, SpanRoles

log = logging.getLogger(__name__)


class Templater(BaseMixin):
    def __init__(self, *, subject_placeholder="[X]", answer_placeholder="[Y]", capitalize: bool = True, **kw):
        super().__init__(**kw)

        self._subject_placeholder = subject_placeholder
        self._answer_placeholder = answer_placeholder
        self.capitalize: bool = capitalize

        # build the search pattern for the placeholders
        self._search_pattern = re.compile(
            "(" + re.escape(subject_placeholder) + "|" + re.escape(answer_placeholder) + ")"
        )

    @staticmethod
    def capitalize_text(text: str) -> str:
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
    ) -> tuple[str, SpanRoles]:
        """Replace all placeholders in the template with the respective values.

        Parameters:
            template: The temaplate string with appropriate placeholders.
            subject: The subject label to fill in at the resective placeholder.
            answer: The answer span to fill in.

        Returns:
            The final string as well as the spans of the respective elements in the final string.
        """
        # initialize the string
        text: str = ""

        # initialize dictionary with final spans
        spans: SpanRoles = {
            "subject": [],
            "answer": [],
        }

        last_match_end: Optional[int] = None

        # count the offset introduced by replacements
        offset = 0

        for match in self._search_pattern.finditer(template):
            key = match.group()

            value: Optional[str]
            span_list: list[Span]

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
                # Replace the placeholder with the value

                # Position in the original template
                start, end = match.span()

                # Add the unmatched text
                text += template[last_match_end : match.start()] + value
                last_match_end = match.end()

                # Add the the added span (value) to the span list
                span_list.append((start + offset, start + offset + len(value)))

                # Shift the offest
                offset += len(value) - (end - start)

        # append the remainder of the template.
        text += template[last_match_end:]

        if self.capitalize:
            text = self.capitalize_text(text)

        return text, spans

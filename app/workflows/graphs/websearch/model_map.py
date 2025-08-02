"""Mapping of tasks to their corresponding LLM model names."""

import enum

from app import settings


class LLMModelMap(str, enum.Enum):
    """Enum mapping specific agent tasks to their respective LLM model identifiers."""

    QUESTION_REWRITER = settings.MODEL_NAME
    QUESTION_ENHANCER = settings.MODEL_NAME
    ANSWER_GENERATOR = settings.MODEL_NAME

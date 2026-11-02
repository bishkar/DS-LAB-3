"""Shared, pydantic-based input validation for all NLP tasks.

Every task validates its inputs through the helpers below instead of repeating
``isinstance`` checks. Pydantic does the type checking (in strict mode, so
``"1"`` is not an int and ``True`` is not a number), and the helpers turn
``ValidationError`` into the ``TypeError`` / ``ValueError`` (with the exact
messages) that the public API documents.
"""
from functools import lru_cache
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, Type

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
)


@lru_cache(maxsize=None)
def _adapter(tp: Any) -> TypeAdapter:
    """Build (and cache) a strict TypeAdapter for ``tp``."""
    return TypeAdapter(tp, config=ConfigDict(strict=True, arbitrary_types_allowed=True))


def check_type(value: Any, tp: Any, message: str, error: Type[Exception] = TypeError) -> Any:
    """Validate ``value`` against ``tp`` and raise ``error(message)`` on mismatch."""
    if isinstance(value, np.integer):  # numpy ints are fine wherever ints are
        value = int(value)
    try:
        return _adapter(tp).validate_python(value)
    except ValidationError as exc:
        raise error(message) from exc


def validate_text(value: Any, label: str) -> str:
    """Require a non-blank string.

    Raises ``TypeError("<label> must be a string")`` or
    ``ValueError("<label> cannot be empty")``.
    """
    text = check_type(value, str, f"{label} must be a string")
    if not text.strip():
        raise ValueError(f"{label} cannot be empty")
    return text


def validate_list(
    value: Any,
    label: str,
    item_type: Any = None,
    item_message: Optional[str] = None,
    item_error: Type[Exception] = ValueError,
) -> list:
    """Require a non-empty list, optionally checking the type of its items.

    Raises ``TypeError("<label> must be a list")``,
    ``ValueError("<label> cannot be empty")`` and, when ``item_type`` is given,
    ``item_error(item_message)`` for a list whose items do not match.
    """
    items = check_type(value, list, f"{label} must be a list")
    if not items:
        raise ValueError(f"{label} cannot be empty")
    if item_type is not None:
        check_type(items, List[item_type], item_message or f"Invalid items in {label}", item_error)
    return value


def validate_string_list(
    value: Any, label: str, items: str, item_error: Type[Exception] = ValueError
) -> List[str]:
    """Require a non-empty list of strings ("All <items> must be strings")."""
    return validate_list(value, label, str, f"All {items} must be strings", item_error)


def validate_index(doc_index: Any, size: int) -> int:
    """Require ``0 <= doc_index < size``; any problem is a ``ValueError``."""
    message = f"doc_index must be between 0 and {size - 1}"
    index = check_type(doc_index, int, message, ValueError)
    if not 0 <= index < size:
        raise ValueError(message)
    return index


def validate_top_n(top_n: Any) -> int:
    """Require a positive int; any problem is a ``ValueError``."""
    top = check_type(top_n, int, "top_n must be a positive integer", ValueError)
    if top < 1:
        raise ValueError("top_n must be a positive integer")
    return top


def raise_as_value_error(exc: ValidationError, messages: dict) -> None:
    """Re-raise a model ``ValidationError`` as ``ValueError``.

    The message is picked by the name of the first failing field, falling back
    to pydantic's own description.
    """
    error = exc.errors()[0]
    field = error["loc"][0] if error["loc"] else None
    raise ValueError(messages.get(field, f"{field}: {error['msg']}")) from exc


class Word2VecParams(BaseModel):
    """Hyper-parameters accepted by :func:`src.task5_word2vec.train_word2vec`."""

    model_config = ConfigDict(strict=True, frozen=True)

    vector_size: int = Field(100, gt=0)
    window: int = Field(5, gt=0)
    min_count: int = Field(1, ge=0)
    workers: int = Field(4, gt=0)


class PipelineConfig(BaseModel):
    """Options accepted by :func:`src.task7_nlp_pipeline.analyze_documents`."""

    model_config = ConfigDict(strict=True, frozen=True)

    use_lemmatization: bool = True
    ngram_range: Tuple[int, int] = (1, 2)
    vector_method: Literal["bow", "tfidf"] = "tfidf"

    @field_validator("ngram_range")
    @classmethod
    def _check_ngram_range(cls, value: Tuple[int, int]) -> Tuple[int, int]:
        low, high = value
        if low < 1 or low > high:
            raise ValueError("expected 1 <= min_n <= max_n")
        return value

    ERROR_MESSAGES: ClassVar[Dict[str, str]] = {
        "use_lemmatization": "use_lemmatization must be a boolean",
        "ngram_range": "Invalid ngram_range: expected a tuple (min_n, max_n) with 1 <= min_n <= max_n",
        "vector_method": "vector_method must be 'bow' or 'tfidf'",
    }

    @classmethod
    def build(cls, **options: Any) -> "PipelineConfig":
        """Create a config, reporting problems as ``ValueError``."""
        try:
            return cls(**options)
        except ValidationError as exc:
            raise_as_value_error(exc, cls.ERROR_MESSAGES)

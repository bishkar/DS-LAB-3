"""Task 1: Text Cleaning and Tokenization."""
import re
from typing import List

from src.utils import tokenize
from src.validation import validate_text


def clean_and_tokenize(text: str) -> List[str]:
    """
    Clean text by removing punctuation, converting to lowercase, and tokenizing.

    Args:
        text (str): The input text string to clean and tokenize.

    Returns:
        List[str]: A list of tokens (words).

    Raises:
        ValueError: If text is None or empty.
        TypeError: If text is not a string.

    Example:
        >>> clean_and_tokenize("Hello, World!")
        ['hello', 'world']
    """
    validate_text(text, "Input text")

    cleaned = re.sub(r"[^\w\s]|_", " ", text).lower()
    return tokenize(cleaned)


if __name__ == "__main__":
    example_text = (
        "Cleans the input text by removing punctuation and converting it to lowercase, "
        "then tokenizes it into individual words."
    )
    print(clean_and_tokenize(example_text))


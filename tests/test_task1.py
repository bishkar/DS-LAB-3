"""Tests for Task 1: Text Cleaning and Tokenization."""
import pytest
from src.task1_text_cleaning import clean_and_tokenize


def test_clean_and_tokenize_basic():
    """Test basic text cleaning and tokenization."""
    text = "Hello, World! NLP is amazing."
    expected = ['hello', 'world', 'nlp', 'is', 'amazing']
    result = clean_and_tokenize(text)
    assert result == expected, f"Expected {expected} but got {result}"


def test_clean_and_tokenize_with_numbers():
    """Test that numbers are preserved."""
    text = "Python 3 is better than Python 2"
    result = clean_and_tokenize(text)
    assert '3' in result, "Numbers should be preserved"
    assert '2' in result, "Numbers should be preserved"


def test_clean_and_tokenize_multiple_punctuation():
    """Test removal of multiple punctuation marks."""
    text = "Hello!!! How are you???"
    result = clean_and_tokenize(text)
    assert all(char.isalnum() or char.isspace() for token in result for char in token), \
        "All punctuation should be removed"


def test_clean_and_tokenize_empty_string():
    """Test that empty string raises ValueError."""
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        clean_and_tokenize("")


def test_clean_and_tokenize_whitespace_only():
    """Test that whitespace-only string raises ValueError."""
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        clean_and_tokenize("   ")


def test_clean_and_tokenize_non_string():
    """Test that non-string input raises TypeError."""
    with pytest.raises(TypeError, match="Input text must be a string"):
        clean_and_tokenize(123)


def test_clean_and_tokenize_lowercase():
    """Test that all text is converted to lowercase."""
    text = "HELLO World"
    result = clean_and_tokenize(text)
    assert all(token.islower() or not token.isalpha() for token in result), \
        "All alphabetic tokens should be lowercase"



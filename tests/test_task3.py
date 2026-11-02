"""Tests for Task 3: Bag of Words (BoW) Implementation."""
import pytest
import pandas as pd
from src.task3_BoW import bag_of_words


def test_bag_of_words_column_match():
    """Test that BoW generates correct columns."""
    texts = ["NLP is fun", "NLP is challenging"]
    result = bag_of_words(texts)
    expected_columns = {'nlp', 'is', 'fun', 'challenging'}
    assert set(result.columns) == expected_columns, \
        f"BoW column mismatch. Expected {expected_columns}, got {set(result.columns)}"


def test_bag_of_words_shape_match():
    """Test that BoW output has correct shape."""
    texts = ["NLP is fun", "NLP is challenging"]
    result = bag_of_words(texts)
    assert result.shape == (2, 4), \
        f"BoW shape mismatch. Expected (2, 4), got {result.shape}"


def test_bag_of_words_frequency_count():
    """Test that word frequencies are counted correctly."""
    texts = ["NLP NLP is fun", "NLP is challenging"]
    result = bag_of_words(texts)
    assert result.loc[0, 'nlp'] == 2, "Word 'nlp' should appear twice in first text"
    assert result.loc[1, 'nlp'] == 1, "Word 'nlp' should appear once in second text"


def test_bag_of_words_returns_dataframe():
    """Test that function returns a pandas DataFrame."""
    texts = ["test"]
    result = bag_of_words(texts)
    assert isinstance(result, pd.DataFrame), "Result should be a pandas DataFrame"


def test_bag_of_words_empty_list():
    """Test that empty list raises ValueError."""
    with pytest.raises(ValueError, match="Input texts cannot be empty"):
        bag_of_words([])


def test_bag_of_words_non_list():
    """Test that non-list input raises TypeError."""
    with pytest.raises(TypeError, match="Input texts must be a list"):
        bag_of_words("not a list")


def test_bag_of_words_non_string_elements():
    """Test that list with non-string elements raises ValueError."""
    with pytest.raises(ValueError, match="All texts must be strings"):
        bag_of_words(["text", 123, "more text"])


def test_bag_of_words_single_text():
    """Test processing a single text."""
    texts = ["single text"]
    result = bag_of_words(texts)
    assert result.shape[0] == 1, "Should have one row for single text"


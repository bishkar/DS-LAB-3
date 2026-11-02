"""Tests for Task 4: TF-IDF Calculation."""
import pytest
import pandas as pd
from src.task4_TFIDF import calculate_tfidf


def test_calculate_tfidf_correct_terms():
    """Test that TF-IDF includes expected terms."""
    texts = ["NLP is amazing", "Learning NLP"]
    result = calculate_tfidf(texts)
    assert 'nlp' in result.columns, \
        f"TF-IDF did not include expected term 'nlp'. Columns: {result.columns.tolist()}"


def test_calculate_tfidf_correct_number_rows():
    """Test that TF-IDF has correct number of rows."""
    texts = ["NLP is amazing", "Learning NLP"]
    result = calculate_tfidf(texts)
    assert result.shape[0] == 2, \
        f"TF-IDF incorrect number of rows. Expected 2, got {result.shape[0]}"


def test_calculate_tfidf_returns_dataframe():
    """Test that function returns a pandas DataFrame."""
    texts = ["test"]
    result = calculate_tfidf(texts)
    assert isinstance(result, pd.DataFrame), "Result should be a pandas DataFrame"


def test_calculate_tfidf_values_range():
    """Test that TF-IDF values are in valid range [0, 1]."""
    texts = ["NLP is amazing", "Learning NLP"]
    result = calculate_tfidf(texts)
    assert (result >= 0).all().all(), "TF-IDF values should be non-negative"
    assert (result <= 1).all().all(), "TF-IDF values should not exceed 1"


def test_calculate_tfidf_idf_reduces_common_words():
    """Test that common words have lower TF-IDF scores."""
    texts = ["NLP is amazing", "NLP is great", "NLP is fun"]
    result = calculate_tfidf(texts)
    # 'is' and 'nlp' appear in all documents, should have lower scores
    # compared to unique words in the same document
    assert result.loc[0, 'is'] < result.loc[0, 'amazing'], \
        "Common words should have lower TF-IDF scores"


def test_calculate_tfidf_empty_list():
    """Test that empty list raises ValueError."""
    with pytest.raises(ValueError, match="Input texts cannot be empty"):
        calculate_tfidf([])


def test_calculate_tfidf_non_list():
    """Test that non-list input raises TypeError."""
    with pytest.raises(TypeError, match="Input texts must be a list"):
        calculate_tfidf("not a list")


def test_calculate_tfidf_non_string_elements():
    """Test that list with non-string elements raises ValueError."""
    with pytest.raises(ValueError, match="All texts must be strings"):
        calculate_tfidf(["text", 123])

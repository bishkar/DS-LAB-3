"""Tests for Task 2: Stemming and Lemmatization."""
import pytest
from src.task2_stemm_lemma import stem_and_lemmatize


def test_stem_basic():
    """Test basic stemming functionality."""
    tokens = ['running', 'jumps', 'easily', 'cats']
    result = stem_and_lemmatize(tokens)
    assert result['stemmed'] == ['run', 'jump', 'easili', 'cat'], \
        f"Stemming failed. Expected ['run', 'jump', 'easili', 'cat'], got {result['stemmed']}"


def test_lemmatize_basic():
    """Test basic lemmatization functionality."""
    tokens = ['running', 'jumps', 'easily', 'cats']
    result = stem_and_lemmatize(tokens)
    assert result['lemmatized'] == ['running', 'jump', 'easily', 'cat'], \
        f"Lemmatization failed. Expected ['running', 'jump', 'easily', 'cat'], got {result['lemmatized']}"


def test_stem_and_lemmatize_original_preserved():
    """Test that original tokens are preserved in result."""
    tokens = ['running', 'jumps']
    result = stem_and_lemmatize(tokens)
    assert result['original'] == tokens, "Original tokens should be preserved"


def test_stem_and_lemmatize_return_keys():
    """Test that result contains all required keys."""
    tokens = ['test']
    result = stem_and_lemmatize(tokens)
    assert 'original' in result, "Result should contain 'original' key"
    assert 'stemmed' in result, "Result should contain 'stemmed' key"
    assert 'lemmatized' in result, "Result should contain 'lemmatized' key"


def test_stem_and_lemmatize_empty_list():
    """Test that empty list raises ValueError."""
    with pytest.raises(ValueError, match="Input tokens cannot be empty"):
        stem_and_lemmatize([])


def test_stem_and_lemmatize_non_list():
    """Test that non-list input raises TypeError."""
    with pytest.raises(TypeError, match="Input tokens must be a list"):
        stem_and_lemmatize("not a list")


def test_stem_and_lemmatize_non_string_tokens():
    """Test that list with non-string elements raises ValueError."""
    with pytest.raises(ValueError, match="All tokens must be strings"):
        stem_and_lemmatize(['word', 123, 'another'])


def test_stem_and_lemmatize_single_token():
    """Test processing a single token."""
    tokens = ['running']
    result = stem_and_lemmatize(tokens)
    assert len(result['stemmed']) == 1, "Should process single token"
    assert len(result['lemmatized']) == 1, "Should process single token"


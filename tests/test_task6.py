"""Tests for Task 6: N-gram Generation and Frequency Counting."""
import pytest
from src.task6_Ngrams import count_ngrams, generate_ngrams


def test_generate_ngrams_bigrams():
    """Test generation of bigrams."""
    text = "I love NLP and learning NLP techniques"
    n = 2
    expected_bigrams = [
        ('I', 'love'),
        ('love', 'NLP'),
        ('NLP', 'and'),
        ('and', 'learning'),
        ('learning', 'NLP'),
        ('NLP', 'techniques')
    ]
    result = generate_ngrams(text, n)
    assert result == expected_bigrams, \
        f"Expected {expected_bigrams} but got {result}"


def test_generate_ngrams_trigrams():
    """Test generation of trigrams."""
    text = "I love NLP"
    n = 3
    expected_trigrams = [('I', 'love', 'NLP')]
    result = generate_ngrams(text, n)
    assert result == expected_trigrams, \
        f"Expected {expected_trigrams} but got {result}"


def test_generate_ngrams_unigrams():
    """Test generation of unigrams."""
    text = "I love NLP"
    n = 1
    expected_unigrams = [('I',), ('love',), ('NLP',)]
    result = generate_ngrams(text, n)
    assert result == expected_unigrams, \
        f"Expected {expected_unigrams} but got {result}"


def test_count_ngrams_frequency():
    """Test counting of N-gram frequencies."""
    ngrams = [
        ('I', 'love'),
        ('love', 'NLP'),
        ('NLP', 'and'),
        ('and', 'learning'),
        ('learning', 'NLP'),
        ('NLP', 'techniques')
    ]
    expected_counts = {
        ('I', 'love'): 1,
        ('love', 'NLP'): 1,
        ('NLP', 'and'): 1,
        ('and', 'learning'): 1,
        ('learning', 'NLP'): 1,
        ('NLP', 'techniques'): 1
    }
    result = count_ngrams(ngrams)
    assert result == expected_counts, \
        f"Expected {expected_counts} but got {result}"


def test_count_ngrams_with_duplicates():
    """Test counting with duplicate N-grams."""
    ngrams = [('I', 'love'), ('love', 'NLP'), ('I', 'love')]
    result = count_ngrams(ngrams)
    assert result[('I', 'love')] == 2, \
        "N-gram ('I', 'love') should appear twice"
    assert result[('love', 'NLP')] == 1, \
        "N-gram ('love', 'NLP') should appear once"


def test_count_ngrams_returns_dict():
    """Test that count_ngrams returns a dictionary."""
    ngrams = [('test', 'ngram')]
    result = count_ngrams(ngrams)
    assert isinstance(result, dict), "Result should be a dictionary"


def test_generate_ngrams_empty_text():
    """Test that empty text raises ValueError."""
    with pytest.raises(ValueError, match="Text cannot be empty"):
        generate_ngrams("", 2)


def test_generate_ngrams_whitespace_only():
    """Test that whitespace-only text raises ValueError."""
    with pytest.raises(ValueError, match="Text cannot be empty"):
        generate_ngrams("   ", 2)


def test_generate_ngrams_invalid_n():
    """Test that invalid n value raises ValueError."""
    with pytest.raises(ValueError, match="N must be at least 1"):
        generate_ngrams("test text", 0)


def test_generate_ngrams_non_string():
    """Test that non-string text raises TypeError."""
    with pytest.raises(TypeError, match="Text must be a string"):
        generate_ngrams(123, 2)


def test_generate_ngrams_non_integer_n():
    """Test that non-integer n raises TypeError."""
    with pytest.raises(TypeError, match="N must be an integer"):
        generate_ngrams("test text", 2.5)


def test_count_ngrams_empty_list():
    """Test that empty N-grams list raises ValueError."""
    with pytest.raises(ValueError, match="N-grams list cannot be empty"):
        count_ngrams([])


def test_count_ngrams_non_list():
    """Test that non-list input raises TypeError."""
    with pytest.raises(TypeError, match="N-grams list must be a list"):
        count_ngrams("not a list")


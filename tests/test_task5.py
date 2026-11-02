"""Tests for Task 5: Word2Vec Model Training."""
import pytest
from gensim.models import Word2Vec
from src.task5_word2vec import find_similar_words, train_word2vec


def test_train_word2vec_valid_model():
    """Test that Word2Vec model is trained successfully."""
    sentences = [
        ['i', 'love', 'nlp'],
        ['nlp', 'is', 'fun'],
        ['deep', 'learning', 'is', 'a', 'subset', 'of', 'machine', 'learning'],
        ['machine', 'learning', 'is', 'fun']
    ]
    model = train_word2vec(sentences)
    assert model is not None, "Word2Vec model training failed"
    assert isinstance(model, Word2Vec), "Result should be a Word2Vec model"


def test_train_word2vec_valid_word_found():
    """Test that trained model contains expected words in vocabulary."""
    sentences = [
        ['i', 'love', 'nlp'],
        ['nlp', 'is', 'fun'],
        ['deep', 'learning', 'is', 'a', 'subset', 'of', 'machine', 'learning'],
        ['machine', 'learning', 'is', 'fun']
    ]
    model = train_word2vec(sentences)
    assert 'nlp' in model.wv, "Word 'nlp' not found in vocabulary"
    assert 'learning' in model.wv, "Word 'learning' not found in vocabulary"


def test_find_similar_words_similar_word_found():
    """Test that similar words are found for a given word."""
    sentences = [
        ['i', 'love', 'nlp'],
        ['nlp', 'is', 'fun'],
        ['deep', 'learning', 'is', 'a', 'subset', 'of', 'machine', 'learning'],
        ['machine', 'learning', 'is', 'fun']
    ]
    model = train_word2vec(sentences)
    similar_words = find_similar_words(model, 'nlp')
    assert len(similar_words) > 0, "No similar words found"


def test_find_similar_words_list():
    """Test that find_similar_words returns a list."""
    sentences = [
        ['i', 'love', 'nlp'],
        ['nlp', 'is', 'fun'],
        ['deep', 'learning', 'is', 'a', 'subset', 'of', 'machine', 'learning'],
        ['machine', 'learning', 'is', 'fun']
    ]
    model = train_word2vec(sentences)
    similar_words = find_similar_words(model, 'nlp')
    assert isinstance(similar_words, list), "Expected a list of similar words"


def test_find_similar_words_tuple_format():
    """Test that similar words are returned as tuples with word and score."""
    sentences = [
        ['i', 'love', 'nlp'],
        ['nlp', 'is', 'fun'],
        ['deep', 'learning', 'is', 'a', 'subset', 'of', 'machine', 'learning'],
        ['machine', 'learning', 'is', 'fun']
    ]
    model = train_word2vec(sentences)
    similar_words = find_similar_words(model, 'nlp')
    for item in similar_words:
        assert isinstance(item, tuple), "Each item should be a tuple"
        assert len(item) == 2, "Each tuple should have word and score"
        assert isinstance(item[0], str), "First element should be a word (string)"
        assert isinstance(item[1], float), "Second element should be a score (float)"


def test_find_similar_words_topn_parameter():
    """Test that topn parameter limits the number of results."""
    sentences = [
        ['i', 'love', 'nlp'],
        ['nlp', 'is', 'fun'],
        ['deep', 'learning', 'is', 'a', 'subset', 'of', 'machine', 'learning'],
        ['machine', 'learning', 'is', 'fun']
    ]
    model = train_word2vec(sentences)
    similar_words = find_similar_words(model, 'nlp', topn=3)
    assert len(similar_words) <= 3, "Should return at most 3 similar words"


def test_find_similar_words_word_not_in_vocab():
    """Test that KeyError is raised for word not in vocabulary."""
    sentences = [['i', 'love', 'nlp']]
    model = train_word2vec(sentences)
    with pytest.raises(KeyError, match="not in vocabulary"):
        find_similar_words(model, 'unknown')


def test_train_word2vec_empty_sentences():
    """Test that empty sentences list raises ValueError."""
    with pytest.raises(ValueError, match="Sentences cannot be empty"):
        train_word2vec([])


def test_train_word2vec_non_list():
    """Test that non-list input raises TypeError."""
    with pytest.raises(TypeError, match="Sentences must be a list"):
        train_word2vec("not a list")


def test_train_word2vec_custom_parameters():
    """Test training with custom parameters."""
    sentences = [['i', 'love', 'nlp'], ['nlp', 'is', 'fun']]
    model = train_word2vec(sentences, vector_size=50, window=3)
    assert model.wv.vector_size == 50, "Vector size should be 50"
    assert model.window == 3, "Window size should be 3"


"""Tests for Task 7: Integrated NLP Pipeline."""
import pytest
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

from src.task7_nlp_pipeline import (
    analyze_documents,
    find_similar_documents,
    get_document_keywords,
    compare_vectorization_methods
)


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        "Natural language processing is amazing",
        "Machine learning and NLP are fun",
        "Deep learning models are powerful"
    ]


@pytest.fixture
def analysis_results(sample_documents):
    """Provide analysis results for testing."""
    return analyze_documents(sample_documents, vector_method="tfidf")


def test_analyze_documents_basic(sample_documents):
    """Test basic document analysis."""
    results = analyze_documents(sample_documents)
    
    assert isinstance(results, dict), "Results should be a dictionary"
    assert "preprocessed" in results, "Results should contain preprocessed section"
    assert "vectorization" in results, "Results should contain vectorization section"
    assert "word_embeddings" in results, "Results should contain word_embeddings section"
    assert "ngram_analysis" in results, "Results should contain ngram_analysis section"
    assert "document_similarity" in results, "Results should contain document_similarity section"


def test_analyze_documents_preprocessed_structure(analysis_results):
    """Test preprocessed results structure."""
    preprocessed = analysis_results["preprocessed"]
    
    assert "documents" in preprocessed, "Should contain documents"
    assert "total_tokens" in preprocessed, "Should contain total_tokens"
    assert "unique_tokens" in preprocessed, "Should contain unique_tokens"
    assert "avg_doc_length" in preprocessed, "Should contain avg_doc_length"
    
    assert isinstance(preprocessed["documents"], list), "Documents should be a list"
    assert isinstance(preprocessed["total_tokens"], int), "total_tokens should be int"
    assert isinstance(preprocessed["unique_tokens"], int), "unique_tokens should be int"
    assert isinstance(preprocessed["avg_doc_length"], float), "avg_doc_length should be float"


def test_analyze_documents_vectorization_structure(analysis_results):
    """Test vectorization results structure."""
    vectorization = analysis_results["vectorization"]
    
    assert "method" in vectorization, "Should contain method"
    assert "feature_matrix" in vectorization, "Should contain feature_matrix"
    assert "feature_names" in vectorization, "Should contain feature_names"
    assert "shape" in vectorization, "Should contain shape"
    
    assert vectorization["method"] in ["bow", "tfidf"], "Method should be bow or tfidf"
    assert isinstance(vectorization["feature_matrix"], pd.DataFrame), \
        "feature_matrix should be DataFrame"
    assert isinstance(vectorization["feature_names"], list), \
        "feature_names should be list"


def test_analyze_documents_word_embeddings_structure(analysis_results):
    """Test word embeddings results structure."""
    embeddings = analysis_results["word_embeddings"]
    
    assert "model" in embeddings, "Should contain model"
    assert "vocabulary_size" in embeddings, "Should contain vocabulary_size"
    assert "vector_size" in embeddings, "Should contain vector_size"
    
    assert isinstance(embeddings["model"], Word2Vec) or embeddings["model"] is None, \
        "model should be Word2Vec or None"
    assert isinstance(embeddings["vocabulary_size"], int), \
        "vocabulary_size should be int"
    assert isinstance(embeddings["vector_size"], int), \
        "vector_size should be int"


def test_analyze_documents_ngram_structure(analysis_results):
    """Test n-gram analysis results structure."""
    ngrams = analysis_results["ngram_analysis"]
    
    assert "top_bigrams" in ngrams, "Should contain top_bigrams"
    assert "top_trigrams" in ngrams, "Should contain top_trigrams"
    assert "bigram_diversity" in ngrams, "Should contain bigram_diversity"
    
    assert isinstance(ngrams["top_bigrams"], list), "top_bigrams should be list"
    assert isinstance(ngrams["top_trigrams"], list), "top_trigrams should be list"
    assert isinstance(ngrams["bigram_diversity"], float), \
        "bigram_diversity should be float"


def test_analyze_documents_similarity_matrix(analysis_results):
    """Test similarity matrix properties."""
    similarity = analysis_results["document_similarity"]
    
    assert isinstance(similarity, pd.DataFrame), "Should be a DataFrame"
    assert similarity.shape[0] == similarity.shape[1], "Should be square matrix"
    
    # Check diagonal values are 1.0 (self-similarity)
    for i in range(len(similarity)):
        assert abs(similarity.iloc[i, i] - 1.0) < 0.01, \
            f"Diagonal element [{i},{i}] should be 1.0"
    
    # Check symmetry
    for i in range(len(similarity)):
        for j in range(len(similarity)):
            assert abs(similarity.iloc[i, j] - similarity.iloc[j, i]) < 0.01, \
                f"Matrix should be symmetric at [{i},{j}]"


def test_analyze_documents_with_bow(sample_documents):
    """Test analysis with BoW vectorization."""
    results = analyze_documents(sample_documents, vector_method="bow")
    
    assert results["vectorization"]["method"] == "bow", \
        "Method should be bow"
    
    # BoW values should be integers (counts)
    feature_matrix = results["vectorization"]["feature_matrix"]
    assert (feature_matrix >= 0).all().all(), "BoW values should be non-negative"


def test_analyze_documents_with_tfidf(sample_documents):
    """Test analysis with TF-IDF vectorization."""
    results = analyze_documents(sample_documents, vector_method="tfidf")
    
    assert results["vectorization"]["method"] == "tfidf", \
        "Method should be tfidf"
    
    # TF-IDF values should be in range [0, 1]
    feature_matrix = results["vectorization"]["feature_matrix"]
    assert (feature_matrix >= 0).all().all(), "TF-IDF values should be non-negative"
    assert (feature_matrix <= 1).all().all(), "TF-IDF values should not exceed 1"


def test_analyze_documents_lemmatization_vs_stemming(sample_documents):
    """Test difference between lemmatization and stemming."""
    lemma_results = analyze_documents(sample_documents, use_lemmatization=True)
    stem_results = analyze_documents(sample_documents, use_lemmatization=False)
    
    # Both should produce results
    assert lemma_results is not None, "Lemmatization should produce results"
    assert stem_results is not None, "Stemming should produce results"
    
    # Token counts might differ slightly
    lemma_tokens = lemma_results["preprocessed"]["total_tokens"]
    stem_tokens = stem_results["preprocessed"]["total_tokens"]
    assert lemma_tokens > 0 and stem_tokens > 0, "Both should produce tokens"


def test_analyze_documents_custom_ngram_range(sample_documents):
    """Test analysis with custom n-gram range."""
    results = analyze_documents(sample_documents, ngram_range=(1, 3))
    
    # Should include unigrams, bigrams, and trigrams
    feature_names = results["vectorization"]["feature_names"]
    assert len(feature_names) > 0, "Should have features"


def test_analyze_documents_empty_list():
    """Test that empty document list raises ValueError."""
    with pytest.raises(ValueError, match="Documents cannot be empty"):
        analyze_documents([])


def test_analyze_documents_single_document():
    """Test that single document raises ValueError."""
    with pytest.raises(ValueError, match="At least 2 documents are required"):
        analyze_documents(["Single document"])


def test_analyze_documents_non_list():
    """Test that non-list input raises TypeError."""
    with pytest.raises(TypeError, match="Documents must be a list"):
        analyze_documents("not a list")


def test_analyze_documents_non_string_elements():
    """Test that non-string elements raise TypeError."""
    with pytest.raises(TypeError, match="All documents must be strings"):
        analyze_documents(["Valid string", 123, "Another string"])


def test_analyze_documents_invalid_vector_method(sample_documents):
    """Test that invalid vector method raises ValueError."""
    with pytest.raises(ValueError, match="vector_method must be"):
        analyze_documents(sample_documents, vector_method="invalid")


def test_analyze_documents_invalid_ngram_range(sample_documents):
    """Test that invalid n-gram range raises ValueError."""
    with pytest.raises(ValueError, match="Invalid ngram_range"):
        analyze_documents(sample_documents, ngram_range=(3, 1))


def test_find_similar_documents_basic(analysis_results):
    """Test finding similar documents."""
    similarity_matrix = analysis_results["document_similarity"]
    similar = find_similar_documents(0, similarity_matrix, top_n=2)
    
    assert isinstance(similar, list), "Should return a list"
    assert len(similar) <= 2, "Should return at most top_n documents"
    
    # Each item should be a tuple of (index, score)
    for item in similar:
        assert isinstance(item, tuple), "Each item should be a tuple"
        assert len(item) == 2, "Each tuple should have 2 elements"
        assert isinstance(item[0], int), "First element should be int (index)"
        assert isinstance(item[1], float), "Second element should be float (score)"


def test_find_similar_documents_scores_ordered(analysis_results):
    """Test that similarity scores are in descending order."""
    similarity_matrix = analysis_results["document_similarity"]
    similar = find_similar_documents(0, similarity_matrix, top_n=2)
    
    if len(similar) > 1:
        for i in range(len(similar) - 1):
            assert similar[i][1] >= similar[i+1][1], \
                "Scores should be in descending order"


def test_find_similar_documents_excludes_self(analysis_results):
    """Test that the document itself is excluded from results."""
    similarity_matrix = analysis_results["document_similarity"]
    similar = find_similar_documents(0, similarity_matrix, top_n=5)
    
    # Document 0 should not be in the results
    indices = [idx for idx, _ in similar]
    assert 0 not in indices, "Should exclude the document itself"


def test_find_similar_documents_invalid_index(analysis_results):
    """Test that invalid document index raises ValueError."""
    similarity_matrix = analysis_results["document_similarity"]
    
    with pytest.raises(ValueError, match="doc_index must be between"):
        find_similar_documents(99, similarity_matrix)


def test_find_similar_documents_non_dataframe():
    """Test that non-DataFrame input raises TypeError."""
    with pytest.raises(TypeError, match="similarity_matrix must be a pandas DataFrame"):
        find_similar_documents(0, "not a dataframe")


def test_get_document_keywords_basic(analysis_results):
    """Test extracting keywords from a document."""
    feature_matrix = analysis_results["vectorization"]["feature_matrix"]
    keywords = get_document_keywords(0, feature_matrix, top_n=5)
    
    assert isinstance(keywords, list), "Should return a list"
    assert len(keywords) <= 5, "Should return at most top_n keywords"
    
    # Each item should be a tuple of (keyword, score)
    for item in keywords:
        assert isinstance(item, tuple), "Each item should be a tuple"
        assert len(item) == 2, "Each tuple should have 2 elements"
        assert isinstance(item[0], str), "First element should be str (keyword)"
        assert isinstance(item[1], float), "Second element should be float (score)"


def test_get_document_keywords_scores_ordered(analysis_results):
    """Test that keyword scores are in descending order."""
    feature_matrix = analysis_results["vectorization"]["feature_matrix"]
    keywords = get_document_keywords(0, feature_matrix, top_n=5)
    
    if len(keywords) > 1:
        for i in range(len(keywords) - 1):
            assert keywords[i][1] >= keywords[i+1][1], \
                "Scores should be in descending order"


def test_get_document_keywords_invalid_index(analysis_results):
    """Test that invalid document index raises ValueError."""
    feature_matrix = analysis_results["vectorization"]["feature_matrix"]
    
    with pytest.raises(ValueError, match="doc_index must be between"):
        get_document_keywords(99, feature_matrix)


def test_get_document_keywords_non_dataframe():
    """Test that non-DataFrame input raises TypeError."""
    with pytest.raises(TypeError, match="feature_matrix must be a pandas DataFrame"):
        get_document_keywords(0, "not a dataframe")


def test_compare_vectorization_methods_basic(sample_documents):
    """Test comparison of BoW and TF-IDF methods."""
    comparison = compare_vectorization_methods(sample_documents)
    
    assert isinstance(comparison, dict), "Should return a dictionary"
    assert "bow" in comparison, "Should contain bow results"
    assert "tfidf" in comparison, "Should contain tfidf results"
    
    assert isinstance(comparison["bow"], pd.DataFrame), "BoW should be DataFrame"
    assert isinstance(comparison["tfidf"], pd.DataFrame), "TF-IDF should be DataFrame"


def test_compare_vectorization_methods_same_shape(sample_documents):
    """Test that BoW and TF-IDF have the same shape."""
    comparison = compare_vectorization_methods(sample_documents)
    
    bow_shape = comparison["bow"].shape
    tfidf_shape = comparison["tfidf"].shape
    
    assert bow_shape == tfidf_shape, \
        "BoW and TF-IDF should have the same shape"


def test_compare_vectorization_methods_different_values(sample_documents):
    """Test that BoW and TF-IDF produce different values."""
    comparison = compare_vectorization_methods(sample_documents)
    
    bow_matrix = comparison["bow"]
    tfidf_matrix = comparison["tfidf"]
    
    # Values should be different (not identical)
    assert not bow_matrix.equals(tfidf_matrix), \
        "BoW and TF-IDF should produce different values"


def test_compare_vectorization_methods_empty_list():
    """Test that empty document list raises ValueError."""
    with pytest.raises(ValueError, match="Documents cannot be empty"):
        compare_vectorization_methods([])


def test_compare_vectorization_methods_single_document():
    """Test that single document raises ValueError."""
    with pytest.raises(ValueError, match="At least 2 documents are required"):
        compare_vectorization_methods(["Single document"])


def test_analyze_documents_large_corpus():
    """Test analysis with a larger corpus."""
    large_corpus = [
        f"Document number {i} discussing various NLP topics and techniques"
        for i in range(10)
    ]
    
    results = analyze_documents(large_corpus)
    
    assert results["preprocessed"]["total_tokens"] > 0, "Should have tokens"
    assert results["word_embeddings"]["vocabulary_size"] > 0, "Should have vocabulary"
    assert results["vectorization"]["shape"][0] == 10, "Should have 10 documents"


def test_analyze_documents_statistics_consistency(sample_documents):
    """Test that statistics are consistent and logical."""
    results = analyze_documents(sample_documents)
    
    preprocessed = results["preprocessed"]
    
    # Unique tokens should not exceed total tokens
    assert preprocessed["unique_tokens"] <= preprocessed["total_tokens"], \
        "Unique tokens cannot exceed total tokens"
    
    # Average document length should be positive
    assert preprocessed["avg_doc_length"] > 0, \
        "Average document length should be positive"
    
    # Number of documents should match
    assert len(preprocessed["documents"]) == len(sample_documents), \
        "Number of processed documents should match input"


def test_pipeline_integration(sample_documents):
    """Test that all pipeline components work together."""
    results = analyze_documents(sample_documents, use_lemmatization=True)
    
    # Test using results from one component with another
    similarity_matrix = results["document_similarity"]
    feature_matrix = results["vectorization"]["feature_matrix"]
    
    # Find similar documents
    similar = find_similar_documents(0, similarity_matrix, top_n=1)
    assert len(similar) > 0, "Should find similar documents"
    
    # Extract keywords
    keywords = get_document_keywords(0, feature_matrix, top_n=3)
    assert len(keywords) > 0, "Should extract keywords"
    
    # All components should work together seamlessly
    assert results["preprocessed"]["total_tokens"] > 0, "Pipeline should produce tokens"
    assert results["word_embeddings"]["vocabulary_size"] > 0, "Pipeline should build vocabulary"

"""Task 7: Integrated NLP Pipeline."""
from typing import List, Dict, Any, Tuple
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.task1_text_cleaning import clean_and_tokenize
from src.task2_stemm_lemma import stem_and_lemmatize
from src.task5_word2vec import train_word2vec
from src.task6_Ngrams import generate_ngrams, count_ngrams
from src.utils import vectorize
from src.validation import (
    PipelineConfig,
    check_type,
    validate_index,
    validate_string_list,
    validate_top_n,
)

TOP_K_NGRAMS = 10


def _validate_documents(documents: Any) -> None:
    """Require a list of at least 2 strings."""
    validate_string_list(documents, "Documents", "documents", item_error=TypeError)
    if len(documents) < 2:
        raise ValueError("At least 2 documents are required")


def _preprocess(doc: str, use_lemmatization: bool) -> List[str]:
    """Clean, tokenize and normalize one document (blank documents -> [])."""
    try:
        tokens = clean_and_tokenize(doc)
    except ValueError:
        return []
    if not tokens:
        return []
    return stem_and_lemmatize(tokens)["lemmatized" if use_lemmatization else "stemmed"]


def _ngram_stats(docs: List[List[str]], n: int) -> Tuple[List[Tuple[Tuple[str, ...], int]], int, int]:
    """Count n-grams per document; return (top-k, unique count, total count)."""
    all_ngrams = [g for tokens in docs for g in generate_ngrams(" ".join(tokens), n)]
    counts = count_ngrams(all_ngrams) if all_ngrams else {}
    # Most frequent first, ties broken alphabetically for stable output
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_K_NGRAMS]
    return top, len(counts), len(all_ngrams)


def _top_items(series: pd.Series, top_n: int) -> List[Tuple[Any, float]]:
    """Return the ``top_n`` largest (label, value) pairs, values rounded to 4 places."""
    top = series.astype(float).sort_values(ascending=False, kind="stable").head(top_n)
    return [(label, round(float(score), 4)) for label, score in top.items()]


def analyze_documents(
    documents: List[str],
    use_lemmatization: bool = True,
    ngram_range: Tuple[int, int] = (1, 2),
    vector_method: str = "tfidf"
) -> Dict[str, Any]:
    """
    Perform comprehensive NLP analysis on a collection of documents.

    This pipeline integrates text preprocessing, vectorization, word embeddings,
    and n-gram analysis to provide a complete document analysis.

    Args:
        documents (List[str]): List of text documents to analyze.
        use_lemmatization (bool): If True, use lemmatization; else use stemming.
                                  Default is True.
        ngram_range (Tuple[int, int]): Range of n-gram sizes for vectorization.
                                        Default is (1, 2).
        vector_method (str): Vectorization method - "bow" or "tfidf".
                            Default is "tfidf".

    Returns:
        Dict[str, Any]: Comprehensive analysis results including preprocessed
                       documents, vectorization, embeddings, and n-gram analysis.

    Raises:
        TypeError: If documents is not a list or contains non-strings.
        ValueError: If documents is empty, has fewer than 2 documents,
                   vector_method is invalid, or ngram_range is invalid.

    Example:
        >>> docs = ["I love NLP", "NLP is fun", "Machine learning rocks"]
        >>> results = analyze_documents(docs)
        >>> results['preprocessed']['total_tokens']
        9
    """
    # Input validation
    _validate_documents(documents)
    config = PipelineConfig.build(
        use_lemmatization=use_lemmatization,
        ngram_range=ngram_range,
        vector_method=vector_method,
    )

    # Step 1: Preprocess documents
    preprocessed_docs = [_preprocess(doc, config.use_lemmatization) for doc in documents]
    all_tokens = [token for tokens in preprocessed_docs for token in tokens]

    # Calculate preprocessing statistics
    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))
    avg_doc_length = total_tokens / len(preprocessed_docs)

    # Step 2: Vectorization
    # Reconstruct text from processed tokens; keep every token (incl. 1-char ones)
    doc_labels = [f"doc_{i}" for i in range(len(documents))]
    feature_df = vectorize(
        [" ".join(tokens) for tokens in preprocessed_docs],
        config.vector_method,
        index=doc_labels,
        ngram_range=config.ngram_range,
        token_pattern=r"(?u)\b\w+\b",
    )
    feature_names = feature_df.columns

    # Step 3: Word embeddings
    # Filter out empty documents for Word2Vec
    non_empty_docs = [tokens for tokens in preprocessed_docs if tokens]
    word2vec_model = train_word2vec(non_empty_docs) if non_empty_docs else None
    vocabulary_size = len(word2vec_model.wv) if word2vec_model else 0
    vector_size = int(word2vec_model.wv.vector_size) if word2vec_model else 0

    # Step 4: N-gram analysis (n-grams never span document boundaries)
    top_bigrams, unique_bigrams, total_bigrams = _ngram_stats(non_empty_docs, 2)
    top_trigrams, _, _ = _ngram_stats(non_empty_docs, 3)
    bigram_diversity = unique_bigrams / total_bigrams if total_bigrams else 0.0

    # Step 5: Document similarity
    similarity_df = pd.DataFrame(
        cosine_similarity(feature_df.values), index=doc_labels, columns=doc_labels
    )

    # Compile results
    results = {
        "preprocessed": {
            "documents": preprocessed_docs,
            "total_tokens": total_tokens,
            "unique_tokens": unique_tokens,
            "avg_doc_length": round(float(avg_doc_length), 2)
        },
        "vectorization": {
            "method": config.vector_method,
            "feature_matrix": feature_df,
            "feature_names": list(feature_names),
            "shape": feature_df.shape
        },
        "word_embeddings": {
            "model": word2vec_model,
            "vocabulary_size": vocabulary_size,
            "vector_size": vector_size
        },
        "ngram_analysis": {
            "top_bigrams": top_bigrams,
            "top_trigrams": top_trigrams,
            "bigram_diversity": round(float(bigram_diversity), 3)
        },
        "document_similarity": similarity_df
    }
    
    return results


def find_similar_documents(
    doc_index: int,
    similarity_matrix: pd.DataFrame,
    top_n: int = 3
) -> List[Tuple[int, float]]:
    """
    Find the most similar documents to a given document.

    Args:
        doc_index (int): Index of the document to find similarities for.
        similarity_matrix (pd.DataFrame): Document similarity matrix.
        top_n (int): Number of similar documents to return. Default is 3.

    Returns:
        List[Tuple[int, float]]: List of (document_index, similarity_score) tuples,
                                 sorted by similarity in descending order.

    Raises:
        ValueError: If doc_index is out of range or top_n is invalid.
        TypeError: If similarity_matrix is not a DataFrame.

    Example:
        >>> sim_matrix = pd.DataFrame([[1.0, 0.8], [0.8, 1.0]])
        >>> find_similar_documents(0, sim_matrix, top_n=1)
        [(1, 0.8)]
    """
    check_type(similarity_matrix, pd.DataFrame, "similarity_matrix must be a pandas DataFrame")
    doc_index = validate_index(doc_index, len(similarity_matrix))
    top_n = validate_top_n(top_n)

    # Use positions, so the matrix may have any index labels; drop self-similarity
    scores = pd.Series(similarity_matrix.iloc[doc_index].to_numpy())
    return [(int(i), score) for i, score in _top_items(scores.drop(doc_index), top_n)]


def get_document_keywords(
    doc_index: int,
    feature_matrix: pd.DataFrame,
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    Extract top keywords from a document based on feature scores.

    Args:
        doc_index (int): Index of the document to extract keywords from.
        feature_matrix (pd.DataFrame): Feature matrix (BoW or TF-IDF).
        top_n (int): Number of top keywords to return. Default is 5.

    Returns:
        List[Tuple[str, float]]: List of (keyword, score) tuples,
                                 sorted by score in descending order.

    Raises:
        ValueError: If doc_index is out of range or top_n is invalid.
        TypeError: If feature_matrix is not a DataFrame.

    Example:
        >>> features = pd.DataFrame({'nlp': [0.9, 0.2], 'fun': [0.5, 0.8]})
        >>> get_document_keywords(0, features, top_n=2)
        [('nlp', 0.9), ('fun', 0.5)]
    """
    check_type(feature_matrix, pd.DataFrame, "feature_matrix must be a pandas DataFrame")
    doc_index = validate_index(doc_index, len(feature_matrix))
    top_n = validate_top_n(top_n)

    # Only features that actually occur in the document can be its keywords
    scores = feature_matrix.iloc[doc_index]
    return [(str(word), score) for word, score in _top_items(scores[scores > 0], top_n)]


def compare_vectorization_methods(
    documents: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Compare BoW and TF-IDF representations side-by-side.

    Args:
        documents (List[str]): List of text documents to compare.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with 'bow' and 'tfidf' DataFrames.

    Raises:
        TypeError: If documents is not a list or contains non-strings.
        ValueError: If documents is empty or has fewer than 2 documents.

    Example:
        >>> docs = ["I love NLP", "NLP is fun"]
        >>> comparison = compare_vectorization_methods(docs)
        >>> 'bow' in comparison and 'tfidf' in comparison
        True
    """
    _validate_documents(documents)

    # Run analysis with both methods
    bow_results = analyze_documents(documents, vector_method="bow")
    tfidf_results = analyze_documents(documents, vector_method="tfidf")

    return {
        "bow": bow_results['vectorization']['feature_matrix'],
        "tfidf": tfidf_results['vectorization']['feature_matrix']
    }


if __name__ == "__main__":
    # Example usage
    sample_documents = [
        "Natural language processing is amazing and fun to learn.",
        "Machine learning and NLP are related fields in AI.",
        "Deep learning models are powerful for NLP tasks.",
        "Python is great for machine learning and data science."
    ]
    
    print("=" * 80)
    print("INTEGRATED NLP PIPELINE ANALYSIS")
    print("=" * 80)
    
    # Run comprehensive analysis
    results = analyze_documents(
        sample_documents,
        use_lemmatization=True,
        ngram_range=(1, 2),
        vector_method="tfidf"
    )
    
    # Display results
    print("\n1. PREPROCESSING STATISTICS")
    print("-" * 80)
    print(f"Total tokens: {results['preprocessed']['total_tokens']}")
    print(f"Unique tokens: {results['preprocessed']['unique_tokens']}")
    print(f"Average document length: {results['preprocessed']['avg_doc_length']}")
    
    print("\n2. VECTORIZATION")
    print("-" * 80)
    print(f"Method: {results['vectorization']['method'].upper()}")
    print(f"Feature matrix shape: {results['vectorization']['shape']}")
    print(f"Number of features: {len(results['vectorization']['feature_names'])}")
    
    print("\n3. WORD EMBEDDINGS")
    print("-" * 80)
    print(f"Vocabulary size: {results['word_embeddings']['vocabulary_size']}")
    print(f"Vector dimensionality: {results['word_embeddings']['vector_size']}")
    
    print("\n4. N-GRAM ANALYSIS")
    print("-" * 80)
    print(f"Top 5 bigrams:")
    for bigram, count in results['ngram_analysis']['top_bigrams'][:5]:
        print(f"  {bigram}: {count}")
    print(f"Bigram diversity: {results['ngram_analysis']['bigram_diversity']}")
    
    print("\n5. DOCUMENT SIMILARITY")
    print("-" * 80)
    print("Similarity matrix:")
    print(results['document_similarity'])
    
    print("\n6. SIMILAR DOCUMENTS TO DOC_0")
    print("-" * 80)
    similar_docs = find_similar_documents(0, results['document_similarity'], top_n=2)
    for doc_idx, similarity in similar_docs:
        print(f"  Document {doc_idx}: {similarity:.4f}")
    
    print("\n7. TOP KEYWORDS IN DOC_0")
    print("-" * 80)
    keywords = get_document_keywords(0, results['vectorization']['feature_matrix'], top_n=5)
    for keyword, score in keywords:
        print(f"  {keyword}: {score:.4f}")
    
    print("\n8. COMPARISON: BOW vs TF-IDF")
    print("-" * 80)
    comparison = compare_vectorization_methods(sample_documents[:3])
    print("BoW shape:", comparison['bow'].shape)
    print("TF-IDF shape:", comparison['tfidf'].shape)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

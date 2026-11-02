"""Helpers shared by several NLP tasks."""
from typing import List, Optional, Sequence

import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

VECTORIZERS = {"bow": CountVectorizer, "tfidf": TfidfVectorizer}


def tokenize(text: str) -> List[str]:
    """Tokenize with NLTK, falling back to whitespace split without 'punkt' data."""
    try:
        return word_tokenize(text)
    except LookupError:
        return text.split()


def vectorize(
    texts: Sequence[str],
    method: str,
    index: Optional[Sequence[str]] = None,
    **vectorizer_options,
) -> pd.DataFrame:
    """Fit a BoW ("bow") or TF-IDF ("tfidf") vectorizer and return a DataFrame.

    Rows are documents (labelled by ``index`` if given), columns are features.
    """
    vectorizer = VECTORIZERS[method](**vectorizer_options)
    matrix = vectorizer.fit_transform(texts)
    return pd.DataFrame(
        matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=index
    )

"""Task 2: Stemming and Lemmatization."""
from typing import Dict, List
from nltk.stem import PorterStemmer, WordNetLemmatizer


def stem_and_lemmatize(tokens: List[str]) -> Dict[str, List[str]]:
    """
    Apply stemming and lemmatization to input tokens.

    Args:
        tokens (List[str]): A list of tokens (words).

    Returns:
        Dict[str, List[str]]: A dictionary with 'original', 'stemmed', and 
                             'lemmatized' token lists.

    Raises:
        TypeError: If tokens is not a list.
        ValueError: If tokens is empty or contains non-string elements.

    Example:
        >>> stem_and_lemmatize(['running', 'cats'])
        {'original': ['running', 'cats'], 'stemmed': ['run', 'cat'], 
         'lemmatized': ['running', 'cat']}
    """
    pass

    return {
        "original": tokens,
        "stemmed": stemmed,
        "lemmatized": lemmatized
    }


if __name__ == "__main__":
    # Download required NLTK resources (only when running as main)
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    
    example_tokens = ['goes', 'worlds', 'done', 'being', 'undefined']
    result = stem_and_lemmatize(example_tokens)
    print(f"Original: {result['original']}")
    print(f"Stemmed: {result['stemmed']}")
    print(f"Lemmatized: {result['lemmatized']}")

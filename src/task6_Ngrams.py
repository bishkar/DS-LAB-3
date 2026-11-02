"""Task 6: N-gram Generation and Frequency Counting."""
from typing import List, Tuple, Dict
from collections import Counter
from nltk import ngrams
from nltk.tokenize import word_tokenize


def generate_ngrams(text: str, n: int) -> List[Tuple[str, ...]]:
    """
    Generate N-grams from the input text.

    Args:
        text (str): The input text string.
        n (int): The size of the N-grams (e.g., n=2 for bigrams, n=3 for trigrams).

    Returns:
        List[Tuple[str, ...]]: A list of N-grams as tuples.

    Raises:
        TypeError: If text is not a string or n is not an integer.
        ValueError: If text is empty or n is less than 1.

    Example:
        >>> generate_ngrams("I love NLP", 2)
        [('I', 'love'), ('love', 'NLP')]
    """
    pass
    return n_grams


def count_ngrams(ngrams_list: List[Tuple[str, ...]]) -> Dict[Tuple[str, ...], int]:
    """
    Count the frequency of each N-gram in the list.

    Args:
        ngrams_list (List[Tuple[str, ...]]): A list of N-grams.

    Returns:
        Dict[Tuple[str, ...], int]: A dictionary where keys are N-grams and 
                                     values are their frequencies.

    Raises:
        TypeError: If ngrams_list is not a list.
        ValueError: If ngrams_list is empty.

    Example:
        >>> ngrams_list = [('I', 'love'), ('love', 'NLP'), ('I', 'love')]
        >>> count_ngrams(ngrams_list)
        {('I', 'love'): 2, ('love', 'NLP'): 1}
    """
    pass
    return dict(ngram_counts)


if __name__ == "__main__":
    # Download required NLTK resources (only when running as main)
    import nltk
    nltk.download('punkt', quiet=True)

    text = "I love NLP and learning NLP techniques"

    # Generate bigrams
    bigrams = generate_ngrams(text, 2)
    print("Bigrams:", bigrams)

    # Count bigrams
    bigram_counts = count_ngrams(bigrams)
    print("\nBigram counts:", bigram_counts)
    
    # Generate trigrams
    trigrams = generate_ngrams(text, 3)
    print("\nTrigrams:", trigrams)

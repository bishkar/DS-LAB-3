"""Task 3: Bag of Words (BoW) Implementation."""
from typing import List
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(texts: List[str]) -> pd.DataFrame:
    """
    Convert a list of text strings into a Bag of Words representation.

    Args:
        texts (List[str]): A list of text strings.

    Returns:
        pd.DataFrame: A DataFrame representing the Bag of Words model,
                      with words as columns and their frequencies as values.

    Raises:
        TypeError: If texts is not a list.
        ValueError: If texts is empty or contains non-string elements.

    Example:
        >>> texts = ["I love NLP", "NLP is fun"]
        >>> bag_of_words(texts)
           fun  i  is  love  nlp
        0    0  1   0     1    1
        1    1  0   1     0    1
    """
    pass
    return df


if __name__ == "__main__":
    texts = [
        "Converts a list of text strings into a Bag of Words representation",
        "The words that changed the strings",
        "Your duty words of the heart",
        "The Bag full of strings"
    ]
    print(bag_of_words(texts))

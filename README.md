# Practical Labs 3
# Free Text and NLP in Data Science


A comprehensive collection of Natural Language Processing (NLP) tasks focusing on fundamental text processing techniques, vectorization methods, and word embeddings.

---

## **Setup and Installation**

### **Prerequisites**
- Python 3.11+
- Virtual environment (recommended)

### **Installation Steps**

1. Create and activate virtual environment:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data (done automatically when running individual tasks):
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

---

## **Running Tests**

Execute all tests:
```bash
python -m pytest tests/ -v
```

Run specific test file:
```bash
python -m pytest tests/test_task1.py -v
```

Run with coverage:
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

---

## **Project Structure**

```
MLDS_Sprint12/
├── src/
│   ├── task1_text_cleaning.py      # Text cleaning and tokenization
│   ├── task2_stemm_lemma.py        # Stemming and lemmatization
│   ├── task3_BoW.py                # Bag of Words implementation
│   ├── task4_TFIDF.py              # TF-IDF calculation
│   ├── task5_word2vec.py           # Word2Vec training and similarity
│   ├── task6_Ngrams.py             # N-gram generation and counting
│   └── task7_nlp_pipeline.py       # Integrated NLP pipeline (Essential)
├── tests/
│   ├── test_task1.py               # 7 tests for text cleaning
│   ├── test_task2.py               # 8 tests for stemming/lemmatization
│   ├── test_task3.py               # 8 tests for BoW
│   ├── test_task4.py               # 8 tests for TF-IDF
│   ├── test_task5.py               # 10 tests for Word2Vec
│   ├── test_task6.py               # 13 tests for N-grams
│   └── test_task7.py               # 33 tests for NLP pipeline
├── requirements.txt                 # Project dependencies
├── README.md                        # This file
└── venv/                           # Virtual environment (not in git)
```

---

## **Dependencies**

- **pandas** ~=2.2.1 - Data manipulation and DataFrame operations
- **scikit-learn** ~=1.4.2 - BoW and TF-IDF vectorization
- **nltk** ~=3.8.1 - Text tokenization, stemming, and lemmatization
- **gensim** ~=4.3.3 - Word2Vec model training
- **pytest** ~=8.3.2 - Unit testing framework

---

## **Quick Start - Running the Streamlit App**

### **Setup Instructions**

1. **Create and activate a virtual environment:**

   **Windows:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS/Linux:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Streamlit (if not already included):**
   ```bash
   pip install streamlit
   ```

4. **Download required NLTK data:**
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
   ```

5. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

6. **Access the app:**
   - The app will automatically open in your default browser
   - Or navigate manually to: `http://localhost:8501`

### **Using the App**

The Streamlit app provides an interactive interface to demonstrate all 7 NLP tasks:
- Select a task from the dropdown menu
- Enter your own text or use the provided examples
- Click the corresponding button to see results
- Explore different parameters and options for each task

---

## **Task 1: Text Cleaning and Tokenization**

### **Objective**
Implement a function that performs basic text preprocessing by cleaning raw text and splitting it into individual tokens.

### **Function Signature**
```python
def clean_and_tokenize(text: str) -> List[str]
```

### **Requirements**
- Remove all punctuation marks from the input text
- Convert all text to lowercase
- Tokenize the cleaned text into individual words
- Return a list of tokens
- Validate input (raise `TypeError` for non-string input, `ValueError` for empty text)

### **Example**
```python
>>> clean_and_tokenize("Hello, World! NLP is amazing.")
['hello', 'world', 'nlp', 'is', 'amazing']
```

---

## **Task 2: Stemming and Lemmatization**

### **Objective**
Apply both stemming and lemmatization techniques to a list of tokens to reduce words to their root forms.

### **Function Signature**
```python
def stem_and_lemmatize(tokens: List[str]) -> Dict[str, List[str]]
```

### **Requirements**
- Use Porter Stemmer for stemming
- Use WordNet Lemmatizer for lemmatization
- Return a dictionary containing:
  - `"original"`: the input tokens
  - `"stemmed"`: stemmed versions of the tokens
  - `"lemmatized"`: lemmatized versions of the tokens
- Validate input (raise `TypeError` for non-list input, `ValueError` for empty list or non-string elements)

### **Example**
```python
>>> stem_and_lemmatize(['running', 'jumps', 'easily', 'cats'])
{
    'original': ['running', 'jumps', 'easily', 'cats'],
    'stemmed': ['run', 'jump', 'easili', 'cat'],
    'lemmatized': ['running', 'jump', 'easily', 'cat']
}
```

---

## **Task 3: Bag of Words (BoW) Implementation**

### **Objective**
Convert a collection of text documents into a Bag of Words representation using word frequency counts.

### **Function Signature**
```python
def bag_of_words(texts: List[str]) -> pd.DataFrame
```

### **Requirements**
- Use scikit-learn's `CountVectorizer` to create the BoW representation
- Return a pandas DataFrame where:
  - Each row represents a document
  - Each column represents a unique word
  - Cell values represent word frequencies in the document
- Validate input (raise `TypeError` for non-list input, `ValueError` for empty list or non-string elements)

### **Example**
```python
>>> bag_of_words(["NLP is fun", "NLP is challenging"])
   challenging  fun  is  nlp
0            0    1   1    1
1            1    0   1    1
```

---

## **Task 4: TF-IDF Calculation**

### **Objective**
Calculate Term Frequency-Inverse Document Frequency (TF-IDF) scores for a collection of text documents.

### **Function Signature**
```python
def calculate_tfidf(texts: List[str]) -> pd.DataFrame
```

### **Requirements**
- Use scikit-learn's `TfidfVectorizer` to calculate TF-IDF scores
- Return a pandas DataFrame where:
  - Each row represents a document
  - Each column represents a unique term
  - Cell values represent TF-IDF scores (range: 0.0 to 1.0)
- Validate input (raise `TypeError` for non-list input, `ValueError` for empty list or non-string elements)

### **Key Concepts**
- **TF (Term Frequency)**: How often a term appears in a document
- **IDF (Inverse Document Frequency)**: Reduces weight of common terms across documents
- Common words appearing in many documents receive lower TF-IDF scores

### **Example**
```python
>>> calculate_tfidf(["NLP is amazing", "Learning NLP"])
   amazing       is  learning       nlp
0  0.652491  0.652491  0.000000  0.385372
1  0.000000  0.000000  0.707107  0.707107
```

---

## **Task 5: Word2Vec Model Training**

### **Objective**
Train a Word2Vec model to generate word embeddings and find semantically similar words.

### **Function Signatures**
```python
def train_word2vec(
    sentences: List[List[str]], 
    vector_size: int = 100, 
    window: int = 5, 
    min_count: int = 1, 
    workers: int = 4
) -> Word2Vec

def find_similar_words(
    model: Word2Vec, 
    word: str, 
    topn: int = 10
) -> List[Tuple[str, float]]
```

### **Requirements**

#### `train_word2vec`:
- Train a Word2Vec model using gensim
- Parameters:
  - `vector_size`: Dimensionality of word vectors (default: 100)
  - `window`: Context window size (default: 5)
  - `min_count`: Minimum word frequency threshold (default: 1)
  - `workers`: Number of training threads (default: 4)
- Return the trained Word2Vec model
- Validate input (check for valid list of tokenized sentences)

#### `find_similar_words`:
- Find words most similar to a given word using cosine similarity
- Return a list of tuples containing (word, similarity_score)
- Raise `KeyError` if word is not in vocabulary

### **Example**
```python
>>> sentences = [['i', 'love', 'nlp'], ['nlp', 'is', 'fun']]
>>> model = train_word2vec(sentences)
>>> find_similar_words(model, 'nlp', topn=2)
[('fun', 0.0234), ('is', 0.0198)]
```

---

## **Task 6: N-gram Generation and Frequency Counting**

### **Objective**
Generate N-grams from text and count their frequencies to capture word sequence patterns.

### **Function Signatures**
```python
def generate_ngrams(text: str, n: int) -> List[Tuple[str, ...]]

def count_ngrams(ngrams_list: List[Tuple[str, ...]]) -> Dict[Tuple[str, ...], int]
```

### **Requirements**

#### `generate_ngrams`:
- Tokenize the input text
- Generate N-grams of specified size
- Support unigrams (n=1), bigrams (n=2), trigrams (n=3), etc.
- Return a list of N-gram tuples
- Validate input (raise `TypeError` for invalid types, `ValueError` for empty text or n < 1)

#### `count_ngrams`:
- Count the frequency of each N-gram in the list
- Return a dictionary mapping N-grams to their frequencies
- Validate input (raise appropriate errors for invalid input)

### **Examples**
```python
>>> generate_ngrams("I love NLP", 2)
[('I', 'love'), ('love', 'NLP')]

>>> count_ngrams([('I', 'love'), ('love', 'NLP'), ('I', 'love')])
{('I', 'love'): 2, ('love', 'NLP'): 1}
```

---

## **Task 7: Integrated NLP Pipeline (Essential Task)**

### **Objective**
Build a comprehensive NLP pipeline that combines multiple techniques from Tasks 1-6 to perform document analysis, similarity comparison, and feature extraction.

### **Function Signature**
```python
def analyze_documents(
    documents: List[str],
    use_lemmatization: bool = True,
    ngram_range: Tuple[int, int] = (1, 2),
    vector_method: str = "tfidf"
) -> Dict[str, Any]
```

### **Requirements**

This pipeline should integrate the following components:

1. **Text Preprocessing** (Task 1 & 2):
   - Clean and tokenize all documents
   - Apply stemming or lemmatization based on parameter
   - Handle edge cases and validate inputs

2. **Feature Extraction** (Tasks 3 & 4):
   - Generate BoW or TF-IDF representation based on `vector_method` parameter
   - Support both unigrams and bigrams (configurable via `ngram_range`)
   - Return feature matrix as DataFrame

3. **Word Embeddings** (Task 5):
   - Train a Word2Vec model on the corpus
   - Extract vocabulary size and vector dimensionality
   - Provide function to find similar words

4. **N-gram Analysis** (Task 6):
   - Extract top-k most frequent bigrams and trigrams
   - Calculate n-gram diversity metrics

### **Return Value Structure**
```python
{
    "preprocessed": {
        "documents": List[List[str]],  # Tokenized and cleaned documents
        "total_tokens": int,
        "unique_tokens": int,
        "avg_doc_length": float
    },
    "vectorization": {
        "method": str,  # "bow" or "tfidf"
        "feature_matrix": pd.DataFrame,
        "feature_names": List[str],
        "shape": Tuple[int, int]
    },
    "word_embeddings": {
        "model": Word2Vec,
        "vocabulary_size": int,
        "vector_size": int
    },
    "ngram_analysis": {
        "top_bigrams": List[Tuple[Tuple[str, str], int]],
        "top_trigrams": List[Tuple[Tuple[str, str, str], int]],
        "bigram_diversity": float  # unique bigrams / total bigrams
    },
    "document_similarity": pd.DataFrame  # Cosine similarity matrix
}
```

### **Implementation Details**

#### **Parameters:**
- `documents` (List[str]): List of text documents to analyze
- `use_lemmatization` (bool): If True, use lemmatization; else use stemming (default: True)
- `ngram_range` (Tuple[int, int]): Range of n-gram sizes for vectorization (default: (1, 2))
- `vector_method` (str): Vectorization method - "bow" or "tfidf" (default: "tfidf")

#### **Validation:**
- Raise `TypeError` if documents is not a list or contains non-strings
- Raise `ValueError` if documents is empty or has fewer than 2 documents
- Raise `ValueError` if `vector_method` not in ["bow", "tfidf"]
- Raise `ValueError` if `ngram_range` is invalid (e.g., min > max)

#### **Additional Functions:**

```python
def find_similar_documents(
    doc_index: int,
    similarity_matrix: pd.DataFrame,
    top_n: int = 3
) -> List[Tuple[int, float]]
    """Find most similar documents to the given document index."""

def get_document_keywords(
    doc_index: int,
    feature_matrix: pd.DataFrame,
    top_n: int = 5
) -> List[Tuple[str, float]]
    """Extract top keywords from a document based on feature scores."""

def compare_vectorization_methods(
    documents: List[str]
) -> Dict[str, pd.DataFrame]
    """Compare BoW and TF-IDF representations side-by-side."""
```

### **Example Usage**

```python
# Sample documents
documents = [
    "Natural language processing is amazing and fun to learn.",
    "Machine learning and NLP are related fields in AI.",
    "Deep learning models are powerful for NLP tasks.",
    "Python is great for machine learning and data science."
]

# Run comprehensive analysis
results = analyze_documents(
    documents, 
    use_lemmatization=True,
    ngram_range=(1, 2),
    vector_method="tfidf"
)

# Access results
print(f"Total tokens: {results['preprocessed']['total_tokens']}")
print(f"Vocabulary size: {results['word_embeddings']['vocabulary_size']}")
print(f"Top bigrams: {results['ngram_analysis']['top_bigrams'][:5]}")

# Find similar documents
similar_docs = find_similar_documents(0, results['document_similarity'], top_n=2)
print(f"Documents similar to doc 0: {similar_docs}")

# Extract keywords
keywords = get_document_keywords(0, results['vectorization']['feature_matrix'])
print(f"Top keywords: {keywords}")
```

### **Expected Output Structure**

```python
{
    "preprocessed": {
        "documents": [
            ['natural', 'language', 'processing', 'amazing', 'fun', 'learn'],
            ['machine', 'learning', 'nlp', 'related', 'field', 'ai'],
            ...
        ],
        "total_tokens": 45,
        "unique_tokens": 28,
        "avg_doc_length": 11.25
    },
    "vectorization": {
        "method": "tfidf",
        "feature_matrix": DataFrame(4x35),  # 4 docs, 35 features
        "feature_names": ['ai', 'amazing', 'deep', ...],
        "shape": (4, 35)
    },
    "word_embeddings": {
        "model": <Word2Vec object>,
        "vocabulary_size": 28,
        "vector_size": 100
    },
    "ngram_analysis": {
        "top_bigrams": [
            (('machine', 'learning'), 2),
            (('natural', 'language'), 1),
            ...
        ],
        "top_trigrams": [...],
        "bigram_diversity": 0.857
    },
    "document_similarity": DataFrame(4x4)  # Cosine similarity scores
}
```

### **Key Concepts Demonstrated**

1. **Pipeline Integration**: Combining multiple NLP techniques into a cohesive workflow
2. **Comparative Analysis**: Understanding trade-offs between BoW and TF-IDF
3. **Document Similarity**: Using vectorization for document comparison
4. **Feature Importance**: Identifying key terms using TF-IDF scores
5. **Corpus Statistics**: Calculating vocabulary diversity and coverage
6. **Scalability**: Designing for larger document collections

---

## **Key NLP Concepts Covered**

1. **Text Preprocessing**: Cleaning, normalization, tokenization
2. **Morphological Analysis**: Stemming vs. lemmatization
3. **Text Vectorization**: BoW and TF-IDF representations
4. **Word Embeddings**: Word2Vec for semantic similarity
5. **N-gram Analysis**: Capturing word sequence patterns
6. **Input Validation**: Robust error handling and type checking
7. **Pipeline Integration**: Building end-to-end NLP analysis systems
8. **Document Similarity**: Cosine similarity and comparative analysis
9. **Feature Extraction**: Identifying important terms and keywords
10. **Corpus Statistics**: Vocabulary diversity and coverage metrics

---

## **Test Coverage**

**Total Tests**: 87 (all passing)

- **Task 1**: 7 tests - Text cleaning and tokenization
- **Task 2**: 8 tests - Stemming and lemmatization  
- **Task 3**: 8 tests - Bag of Words
- **Task 4**: 8 tests - TF-IDF calculation
- **Task 5**: 10 tests - Word2Vec model training
- **Task 6**: 13 tests - N-gram generation
- **Task 7**: 33 tests - Integrated NLP pipeline

Test categories:
- Comprehensive edge case testing
- Input validation tests (TypeError, ValueError)
- Functional correctness tests
- Type checking and format validation
- Integration and pipeline tests

---

## **License**

This project is for educational purposes as part of the Machine Learning and Data Science curriculum.

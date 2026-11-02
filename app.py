"""Streamlit App for NLP Tasks Demonstration."""
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import task functions
from src.task1_text_cleaning import clean_and_tokenize
from src.task2_stemm_lemma import stem_and_lemmatize
from src.task3_BoW import bag_of_words
from src.task4_TFIDF import calculate_tfidf
from src.task5_word2vec import train_word2vec, find_similar_words
from src.task6_Ngrams import generate_ngrams, count_ngrams
from src.task7_nlp_pipeline import analyze_documents, find_similar_documents, get_document_keywords


# Page configuration
st.set_page_config(
    page_title="NLP Tasks Demo",
    page_icon="📚",
    layout="wide"
)

# Title
st.title("📚 NLP Tasks Demonstration")
st.markdown("---")

# Task selection dropdown
task = st.selectbox(
    "Select a Task to Demonstrate:",
    [
        "Task 1: Text Cleaning & Tokenization",
        "Task 2: Stemming & Lemmatization",
        "Task 3: Bag of Words (BoW)",
        "Task 4: TF-IDF Calculation",
        "Task 5: Word2Vec Model",
        "Task 6: N-grams Generation",
        "Task 7: Integrated NLP Pipeline"
    ]
)

st.markdown("---")


# Task 1: Text Cleaning & Tokenization
if task == "Task 1: Text Cleaning & Tokenization":
    st.header("🧹 Task 1: Text Cleaning & Tokenization")
    st.markdown("""
    This task cleans text by removing punctuation, converting to lowercase, 
    and tokenizing into individual words.
    """)
    
    default_text = "Hello, World! This is an example text with PUNCTUATION and numbers 123."
    text_input = st.text_area("Enter text to clean and tokenize:", value=default_text, height=100)
    
    if st.button("Clean and Tokenize", key="task1"):
        try:
            tokens = clean_and_tokenize(text_input)
            st.success("✅ Text cleaned and tokenized successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Text")
                st.write(text_input)
            with col2:
                st.subheader("Tokens")
                st.write(tokens)
                st.info(f"Total tokens: {len(tokens)}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# Task 2: Stemming & Lemmatization
elif task == "Task 2: Stemming & Lemmatization":
    st.header("✂️ Task 2: Stemming & Lemmatization")
    st.markdown("""
    This task applies stemming (reducing words to root form) and lemmatization 
    (converting words to dictionary form) to tokens.
    """)
    
    input_method = st.radio("Input method:", ["Enter text", "Enter tokens directly"])
    
    if input_method == "Enter text":
        default_text = "running cats are playing with toys"
        text_input = st.text_area("Enter text:", value=default_text, height=100)
        
        if st.button("Process Text", key="task2_text"):
            try:
                tokens = clean_and_tokenize(text_input)
                result = stem_and_lemmatize(tokens)
                
                st.success("✅ Text processed successfully!")
                
                df = pd.DataFrame(result)
                st.subheader("Results Comparison")
                st.dataframe(df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        default_tokens = "running, cats, are, playing, with, toys"
        tokens_input = st.text_input("Enter tokens (comma-separated):", value=default_tokens)
        
        if st.button("Process Tokens", key="task2_tokens"):
            try:
                tokens = [t.strip() for t in tokens_input.split(',')]
                result = stem_and_lemmatize(tokens)
                
                st.success("✅ Tokens processed successfully!")
                
                df = pd.DataFrame(result)
                st.subheader("Results Comparison")
                st.dataframe(df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


# Task 3: Bag of Words
elif task == "Task 3: Bag of Words (BoW)":
    st.header("🎒 Task 3: Bag of Words (BoW)")
    st.markdown("""
    This task converts text documents into a Bag of Words representation, 
    showing word frequencies across documents.
    """)
    
    default_texts = """I love NLP
NLP is fun and interesting
Machine learning is amazing
I love machine learning"""
    
    texts_input = st.text_area(
        "Enter texts (one per line):", 
        value=default_texts, 
        height=150
    )
    
    if st.button("Create Bag of Words", key="task3"):
        try:
            texts = [line.strip() for line in texts_input.split('\n') if line.strip()]
            
            if len(texts) < 2:
                st.warning("⚠️ Please enter at least 2 texts.")
            else:
                bow_df = bag_of_words(texts)
                
                st.success("✅ Bag of Words created successfully!")
                st.subheader("Bag of Words Matrix")
                st.dataframe(bow_df, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Documents", len(texts))
                    st.metric("Vocabulary Size", len(bow_df.columns))
                with col2:
                    st.metric("Total Words", int(bow_df.sum().sum()))
                    st.metric("Avg Words per Doc", round(bow_df.sum(axis=1).mean(), 2))
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# Task 4: TF-IDF
elif task == "Task 4: TF-IDF Calculation":
    st.header("📊 Task 4: TF-IDF Calculation")
    st.markdown("""
    This task calculates TF-IDF (Term Frequency-Inverse Document Frequency) scores 
    to identify important words in documents.
    """)
    
    default_texts = """I love NLP
NLP is fun and interesting
Machine learning is amazing
I love machine learning"""
    
    texts_input = st.text_area(
        "Enter texts (one per line):", 
        value=default_texts, 
        height=150
    )
    
    if st.button("Calculate TF-IDF", key="task4"):
        try:
            texts = [line.strip() for line in texts_input.split('\n') if line.strip()]
            
            if len(texts) < 2:
                st.warning("⚠️ Please enter at least 2 texts.")
            else:
                tfidf_df = calculate_tfidf(texts)
                
                st.success("✅ TF-IDF calculated successfully!")
                st.subheader("TF-IDF Matrix")
                st.dataframe(tfidf_df.round(4), use_container_width=True)
                
                # Show top terms per document
                st.subheader("Top Terms per Document")
                for idx, row in tfidf_df.iterrows():
                    top_terms = row.nlargest(3)
                    st.write(f"**Document {idx}:** {', '.join([f'{term} ({score:.3f})' for term, score in top_terms.items()])}")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# Task 5: Word2Vec
elif task == "Task 5: Word2Vec Model":
    st.header("🔤 Task 5: Word2Vec Model")
    st.markdown("""
    This task trains a Word2Vec model to learn word embeddings and find similar words.
    """)
    
    default_sentences = """i love nlp
nlp is fun
deep learning is a subset of machine learning
machine learning is fun
nlp can be challenging
artificial intelligence is related to nlp"""
    
    sentences_input = st.text_area(
        "Enter sentences (one per line, pre-tokenized with spaces):", 
        value=default_sentences, 
        height=150
    )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        vector_size = st.number_input("Vector Size", min_value=10, max_value=300, value=100)
    with col2:
        window = st.number_input("Window Size", min_value=1, max_value=10, value=5)
    with col3:
        min_count = st.number_input("Min Count", min_value=1, max_value=5, value=1)
    with col4:
        topn = st.number_input("Top N Similar Words", min_value=1, max_value=20, value=5)
    
    if st.button("Train Word2Vec Model", key="task5"):
        try:
            sentences = [line.strip().split() for line in sentences_input.split('\n') if line.strip()]
            
            if len(sentences) < 2:
                st.warning("⚠️ Please enter at least 2 sentences.")
            else:
                model = train_word2vec(sentences, vector_size=vector_size, window=window, min_count=min_count)
                
                st.success("✅ Word2Vec model trained successfully!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Vocabulary Size", len(model.wv))
                with col2:
                    st.metric("Vector Dimensions", model.wv.vector_size)
                
                # Find similar words
                st.subheader("Find Similar Words")
                word_input = st.text_input("Enter a word from vocabulary:", value="nlp")
                
                if st.button("Find Similar", key="task5_similar"):
                    try:
                        if word_input in model.wv:
                            similar_words = find_similar_words(model, word_input, topn=topn)
                            
                            st.subheader(f"Words similar to '{word_input}':")
                            df = pd.DataFrame(similar_words, columns=['Word', 'Similarity'])
                            df['Similarity'] = df['Similarity'].round(4)
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.warning(f"⚠️ Word '{word_input}' not in vocabulary.")
                            st.info(f"Available words: {', '.join(list(model.wv.key_to_index.keys())[:20])}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# Task 6: N-grams
elif task == "Task 6: N-grams Generation":
    st.header("🔢 Task 6: N-grams Generation")
    st.markdown("""
    This task generates N-grams (sequences of N words) and counts their frequencies.
    """)
    
    default_text = "I love NLP and learning NLP techniques in machine learning"
    text_input = st.text_area("Enter text:", value=default_text, height=100)
    
    n = st.slider("Select N (for N-grams):", min_value=2, max_value=5, value=2)
    
    if st.button("Generate N-grams", key="task6"):
        try:
            ngrams_list = generate_ngrams(text_input, n)
            
            if ngrams_list:
                ngram_counts = count_ngrams(ngrams_list)
                
                st.success(f"✅ Generated {len(ngrams_list)} {n}-grams!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"Total {n}-grams", len(ngrams_list))
                with col2:
                    st.metric(f"Unique {n}-grams", len(ngram_counts))
                
                # Display top N-grams
                st.subheader(f"Top {n}-grams by Frequency")
                sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)
                
                df = pd.DataFrame(
                    [(' '.join(ngram), count) for ngram, count in sorted_ngrams],
                    columns=[f'{n}-gram', 'Frequency']
                )
                st.dataframe(df, use_container_width=True)
                
                # Display all N-grams
                with st.expander(f"View all {n}-grams"):
                    ngrams_str = [' '.join(ng) for ng in ngrams_list]
                    st.write(ngrams_str)
            else:
                st.warning(f"⚠️ Not enough words to generate {n}-grams.")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# Task 7: Integrated NLP Pipeline
elif task == "Task 7: Integrated NLP Pipeline":
    st.header("🔄 Task 7: Integrated NLP Pipeline")
    st.markdown("""
    This task performs comprehensive NLP analysis including preprocessing, vectorization, 
    word embeddings, n-gram analysis, and document similarity.
    """)
    
    default_docs = """I love natural language processing
NLP is fascinating and fun to learn
Machine learning enables amazing applications
Deep learning is a powerful technique
Artificial intelligence is transforming technology"""
    
    docs_input = st.text_area(
        "Enter documents (one per line):", 
        value=default_docs, 
        height=150
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        use_lemmatization = st.checkbox("Use Lemmatization", value=True)
    with col2:
        vector_method = st.selectbox("Vectorization Method", ["tfidf", "bow"])
    with col3:
        ngram_max = st.number_input("Max N-gram Size", min_value=1, max_value=3, value=2)
    
    if st.button("Run NLP Pipeline", key="task7"):
        try:
            documents = [line.strip() for line in docs_input.split('\n') if line.strip()]
            
            if len(documents) < 2:
                st.warning("⚠️ Please enter at least 2 documents.")
            else:
                with st.spinner("Processing documents..."):
                    results = analyze_documents(
                        documents,
                        use_lemmatization=use_lemmatization,
                        ngram_range=(1, ngram_max),
                        vector_method=vector_method
                    )
                
                st.success("✅ Pipeline completed successfully!")
                
                # Display results in tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📝 Preprocessing", 
                    "📊 Vectorization", 
                    "🔤 Word Embeddings", 
                    "🔢 N-grams", 
                    "🔗 Document Similarity"
                ])
                
                with tab1:
                    st.subheader("Preprocessing Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Tokens", results['preprocessed']['total_tokens'])
                    with col2:
                        st.metric("Unique Tokens", results['preprocessed']['unique_tokens'])
                    with col3:
                        st.metric("Avg Doc Length", results['preprocessed']['avg_doc_length'])
                    
                    st.subheader("Preprocessed Documents")
                    for idx, tokens in enumerate(results['preprocessed']['documents']):
                        with st.expander(f"Document {idx}"):
                            st.write(' '.join(tokens))
                
                with tab2:
                    st.subheader(f"Vectorization ({vector_method.upper()})")
                    st.write(f"**Shape:** {results['vectorization']['shape']}")
                    st.dataframe(results['vectorization']['feature_matrix'].round(4), use_container_width=True)
                    
                    st.subheader("Top Keywords per Document")
                    for idx in range(len(documents)):
                        keywords = get_document_keywords(idx, results['vectorization']['feature_matrix'], top_n=5)
                        st.write(f"**Document {idx}:** {', '.join([f'{word} ({score:.3f})' for word, score in keywords])}")
                
                with tab3:
                    st.subheader("Word Embeddings (Word2Vec)")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Vocabulary Size", results['word_embeddings']['vocabulary_size'])
                    with col2:
                        st.metric("Vector Size", results['word_embeddings']['vector_size'])
                    
                    if results['word_embeddings']['model']:
                        model = results['word_embeddings']['model']
                        word_input = st.text_input("Find similar words for:", value="nlp")
                        
                        if st.button("Find Similar Words", key="task7_similar"):
                            if word_input in model.wv:
                                similar = find_similar_words(model, word_input, topn=5)
                                df = pd.DataFrame(similar, columns=['Word', 'Similarity'])
                                df['Similarity'] = df['Similarity'].round(4)
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.warning(f"Word '{word_input}' not in vocabulary.")
                
                with tab4:
                    st.subheader("N-gram Analysis")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Top Bigrams:**")
                        if results['ngram_analysis']['top_bigrams']:
                            bigrams_df = pd.DataFrame(
                                [(' '.join(bg), count) for bg, count in results['ngram_analysis']['top_bigrams']],
                                columns=['Bigram', 'Frequency']
                            )
                            st.dataframe(bigrams_df, use_container_width=True)
                        else:
                            st.write("No bigrams found.")
                    
                    with col2:
                        st.write("**Top Trigrams:**")
                        if results['ngram_analysis']['top_trigrams']:
                            trigrams_df = pd.DataFrame(
                                [(' '.join(tg), count) for tg, count in results['ngram_analysis']['top_trigrams']],
                                columns=['Trigram', 'Frequency']
                            )
                            st.dataframe(trigrams_df, use_container_width=True)
                        else:
                            st.write("No trigrams found.")
                    
                    st.metric("Bigram Diversity", results['ngram_analysis']['bigram_diversity'])
                
                with tab5:
                    st.subheader("Document Similarity Matrix")
                    st.dataframe(results['document_similarity'].round(4), use_container_width=True)
                    
                    st.subheader("Find Similar Documents")
                    doc_idx = st.number_input("Enter document index:", min_value=0, max_value=len(documents)-1, value=0)
                    
                    if st.button("Find Similar Docs", key="task7_docs"):
                        similar_docs = find_similar_documents(doc_idx, results['document_similarity'], top_n=3)
                        st.write(f"**Documents similar to Document {doc_idx}:**")
                        for sim_idx, sim_score in similar_docs:
                            st.write(f"- Document {sim_idx}: {sim_score:.4f}")
                            st.write(f"  *{documents[sim_idx]}*")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>NLP Tasks Demonstration App | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)

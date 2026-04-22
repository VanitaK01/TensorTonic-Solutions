import numpy as np
from collections import Counter

def tfidf_vectorizer(documents):
    # Handle empty corpus
    if not documents:
        return np.empty((0, 0)), []
    
    # Tokenize documents
    tokenized_docs = []
    for doc in documents:
        if doc.strip() == "":
            tokenized_docs.append([])
        else:
            tokens = doc.lower().split()
            tokenized_docs.append(tokens)
    
    # Build vocabulary (sorted)
    vocab = sorted(set(token for doc in tokenized_docs for token in doc))
    vocab_index = {word: i for i, word in enumerate(vocab)}
    
    n_docs = len(documents)
    n_vocab = len(vocab)
    
    # Initialize TF matrix
    tf = np.zeros((n_docs, n_vocab), dtype=float)
    
    # Compute TF
    for i, doc in enumerate(tokenized_docs):
        if len(doc) == 0:
            continue
        counts = Counter(doc)
        total_terms = len(doc)
        
        for word, count in counts.items():
            j = vocab_index[word]
            tf[i, j] = count / total_terms
    
    # Compute DF (document frequency)
    df = np.zeros(n_vocab, dtype=float)
    for word, j in vocab_index.items():
        df[j] = sum(1 for doc in tokenized_docs if word in doc)
    
    # Compute IDF
    idf = np.log(n_docs / df)
    
    # Compute TF-IDF
    tfidf = tf * idf
    
    return tfidf, vocab
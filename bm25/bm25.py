import numpy as np
from collections import Counter

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    # Handle empty corpus
    if not docs:
        return np.array([], dtype=float)
    
    N = len(docs)
    
    # Document lengths
    doc_lens = np.array([len(doc) for doc in docs], dtype=float)
    avgdl = doc_lens.mean() if N > 0 else 0.0
    
    # Build document frequency (df)
    df = {}
    for doc in docs:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    
    # Precompute IDF for query terms
    idf = {}
    for t in query_tokens:
        if t in df:
            df_t = df[t]
            idf[t] = np.log((N - df_t + 0.5) / (df_t + 0.5) + 1)
        else:
            idf[t] = 0.0  # term not in corpus
    
    # Initialize scores
    scores = np.zeros(N, dtype=float)
    
    # Compute BM25
    for i, doc in enumerate(docs):
        if len(doc) == 0:
            continue
        
        tf_counts = Counter(doc)
        dl = doc_lens[i]
        
        score = 0.0
        
        for t in query_tokens:
            if t not in tf_counts:
                continue
            
            tf_td = tf_counts[t]
            
            numerator = tf_td * (k1 + 1)
            denominator = tf_td + k1 * (1 - b + b * (dl / avgdl))
            
            score += idf[t] * (numerator / denominator)
        
        scores[i] = score
    
    return scores
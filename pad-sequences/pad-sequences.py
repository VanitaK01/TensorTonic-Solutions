import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    # If empty input
    if not seqs:
        return np.empty((0, 0), dtype=int)
    
    # Auto-detect max_len if not given
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    
    # Initialize output array with pad_value
    padded = np.full((len(seqs), max_len), pad_value, dtype=int)
    
    for i, seq in enumerate(seqs):
        # Truncate if longer than max_len
        trunc = seq[:max_len]
        
        # Place sequence (right padding)
        padded[i, :len(trunc)] = trunc
    
    return padded
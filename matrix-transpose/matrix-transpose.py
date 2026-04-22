import numpy as np

def matrix_transpose(A):
    # Handle empty input
    if not A:
        return np.empty((0, 0), dtype=int)
    
    N = len(A)        # number of rows
    M = len(A[0])     # number of columns
    
    # Create result matrix (M x N)
    result = np.zeros((M, N), dtype=int)
    
    # Transpose using indexing
    for i in range(N):
        for j in range(M):
            result[j][i] = A[i][j]
    
    return result
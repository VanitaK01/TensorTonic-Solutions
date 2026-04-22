import numpy as np

def calculate_eigenvalues(matrix):
    # Invalid if empty or not list-like
    if not matrix or not isinstance(matrix, (list, np.ndarray)):
        return None
    
    # Ensure it's a list of lists (or similar)
    try:
        row_lengths = [len(row) for row in matrix]
    except:
        return None
    
    # Check rectangular shape
    if len(set(row_lengths)) != 1:
        return None
    
    # Check square matrix
    n = len(matrix)
    if row_lengths[0] != n:
        return None
    
    # Safe conversion
    A = np.array(matrix, dtype=float)
    
    # Compute eigenvalues
    eigvals = np.linalg.eigvals(A)
    
    # Sort by real, then imaginary
    eigvals_sorted = np.array(sorted(eigvals, key=lambda x: (x.real, x.imag)))
    
    return eigvals_sorted
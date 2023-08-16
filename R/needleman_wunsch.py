import numpy as np
import pandas as pd

def sMatrix(a: list, b: list, filter_func: callable = None) -> np.ndarray:
    """
    Construct a similarity matrix between two lists.
    
    Parameters
    ----------
    a : list
        List a.
    b : list
        List b.
    filter_func : callable, optional
        A similarity function. Default is 1 if equal, -1 otherwise.

    Returns
    -------
    np.ndarray
        A similarity matrix.
    """
    if filter_func is None:
        filter_func = lambda f1, f2: 1 if f1 == f2 else -1
    
    matrix = np.array([[filter_func(b_val, a_val) for a_val in a] for b_val in b])
    return matrix

def alignS(s_matrix: np.ndarray, gap: int = -1) -> np.ndarray:
    """
    Align two sequences using the Needleman-Wunsch algorithm.
    
    Parameters
    ----------
    s_matrix : np.ndarray
        Similarity matrix.
    gap : int, optional
        Penalty assigned to a gap (missing or extra value).

    Returns
    -------
    np.ndarray
        2 column matrix with aligned indices from each sequence.
    """
    s_len = s_matrix.shape[0]
    q_len = s_matrix.shape[1]
    f_matrix = np.zeros((s_len+1, q_len+1))
    
    # Initialize first row and column of f_matrix
    f_matrix[:, 0] = np.arange(0, s_len+1) * gap
    f_matrix[0, :] = np.arange(0, q_len+1) * gap
    
    # Fill the f_matrix
    for i in range(1, s_len+1):
        for j in range(1, q_len+1):
            f_matrix[i, j] = max(f_matrix[i-1, j-1] + s_matrix[i-1, j-1],
                                 f_matrix[i-1, j] + gap,
                                 f_matrix[i, j-1] + gap)
    
    # Backtrace to get aligned sequences
    res = []
    i, j = s_len, q_len
    while i > 0 or j > 0:
        if i > 0 and j > 0 and f_matrix[i, j] == f_matrix[i-1, j-1] + s_matrix[i-1, j-1]:
            res.append([i, j])
            i -= 1
            j -= 1
        elif i > 0 and f_matrix[i, j] == f_matrix[i-1, j] + gap:
            i -= 1
        else:
            j -= 1
    res.reverse()
    return np.array(res)


import numpy as np
import pandas as pd
import numpy as np
from typing import Callable, Sequence, Optional

def sMatrix(a: np.ndarray, b: np.ndarray, filter_func: Optional[Callable[[pd.Timestamp, pd.Timestamp], int]] = None) -> np.ndarray:
    """
    Construct a similarity matrix between two arrays.
    
    Parameters
    ----------
    a : np.ndarray
        First array.
    b : np.ndarray
        Second array.
    filter_func : callable, optional
        A similarity function that takes two timestamps and returns a scalar.
        Default is 1 if timestamps are equal, -1 otherwise.

    Returns
    -------
    np.ndarray
        A similarity matrix with dimensions (len(a), len(b)).
    """
    if filter_func is None:
        filter_func = lambda f1, f2: 1 if f1 == f2 else -1
        
    matrix = np.array([[filter_func(b_val, a_val) for a_val in a] for b_val in b])
    return matrix

def alignS(s_matrix: np.ndarray[int], gap: int = -1) -> np.ndarray[int]:
    """
    Align 2 sequences using the Needleman-Wunsch matching algorithm.

    Parameters
    ----------
    s_matrix : numpy.ndarray[int]
        Similarity matrix with dimensions (len(a), len(b)).
    gap : int, optional
        Penalty assigned to a gap (missing or extra value). Default is -1.

    Returns
    -------
    numpy.ndarray[int]
        2 column matrix. First column corresponds to the rows of the similarity matrix
        (first sequence), Second column corresponds to the columns of the similarity matrix
        (second sequence). Each row maps aligned indices from each sequence.
    """
    
    s_len, q_len = s_matrix.shape
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
import pandas as pd
import numpy as np
from typing import Callable, Tuple, List, Union

def sMatrix(a: List[pd.Timestamp], b: List[pd.Timestamp], filter: Callable = lambda f1, f2: 1 if f1 == f2 else -1) -> np.array:
    """
    Construct a similarity matrix between 2 vectors.
    
    Parameters:
    - a : List
        Vector a.
    - b : List
        Vector b.
    - filter : Callable
        A similarity function. Default: 1 if equal, -1 otherwise.
    
    Returns:
    - np.array
        A similarity matrix.
    """
    return np.array([list(map(lambda x: filter(x, ai), b)) for ai in a])


def alignS(s_matrix: np.array, gap: int = -1) -> pd.DataFrame:
    """
    Align 2 sequences using the Needleman-Wunsch matching algorithm.

    Parameters:
    - s_matrix : np.array
        Similarity matrix.
    - gap : int
        Penalty assigned to a gap (missing or extra value).
    
    Returns:
    - pd.DataFrame
        A 2 column DataFrame with aligned indices from each sequence.
    """
    s_len, q_len = s_matrix.shape
    f = np.zeros((s_len + 1, q_len + 1))
    f[:, 0] = np.arange(s_len + 1) * gap
    f[0, :] = np.arange(q_len + 1) * gap

    for i in range(1, s_len + 1):
        for j in range(1, q_len + 1):
            f[i, j] = max(f[i-1, j-1] + s_matrix[i-1, j-1], f[i-1, j] + gap, f[i, j-1] + gap)

    i, j = s_len, q_len
    res = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and f[i, j] == f[i-1, j-1] + s_matrix[i-1, j-1]:
            res.append([i, j])
            i -= 1
            j -= 1
        elif i > 0 and f[i, j] == f[i-1, j] + gap:
            i -= 1
        else:
            j -= 1

    res_df = pd.DataFrame(res, columns=["a", "b"])
    return res_df.iloc[::-1]

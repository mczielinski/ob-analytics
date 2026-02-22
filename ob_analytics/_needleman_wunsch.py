"""Needleman-Wunsch sequence alignment for bid/ask fill matching.

Adapted from the bioinformatics algorithm to align sequences of
bid and ask fills by timestamp proximity, resolving ambiguous cases
where simple closest-match pairing fails.
"""

import numpy as np
import pandas as pd


def create_similarity_matrix(a: pd.Series, b: pd.Series, cut_off_ms: int) -> np.ndarray:
    """
    Create a similarity matrix based on time differences and cut-off window.

    Parameters
    ----------
    a : pandas.Series
        A pandas Series of timestamps.
    b : pandas.Series
        A pandas Series of timestamps.
    cut_off_ms : int
        The cut-off time window in milliseconds.

    Returns
    -------
    numpy.ndarray
        A NumPy array representing the similarity matrix.
    """
    a_ns = a.values.astype("datetime64[ns]").astype(np.float64)
    b_ns = b.values.astype("datetime64[ns]").astype(np.float64)
    diff_ms = np.abs(a_ns.reshape(-1, 1) - b_ns) / 1e6
    # Avoid divide-by-zero warning: replace zeros before dividing,
    # then overwrite those positions with cut_off_ms via np.where.
    safe_diff = np.where(diff_ms != 0, diff_ms, 1.0)
    return np.where(diff_ms != 0, cut_off_ms / safe_diff, float(cut_off_ms))


def align_sequences(s_matrix: np.ndarray, gap_penalty: int = -1) -> np.ndarray:
    """
    Perform Needleman-Wunsch alignment and return aligned indices.

    Parameters
    ----------
    s_matrix : numpy.ndarray
        The similarity matrix.
    gap_penalty : int, optional
        The penalty for gaps in the alignment. Default is -1.

    Returns
    -------
    numpy.ndarray
        A NumPy array with aligned indices from the two sequences.
    """
    m, n = s_matrix.shape
    f_matrix = np.zeros((m + 1, n + 1))
    f_matrix[0, :] = np.arange(n + 1) * gap_penalty
    f_matrix[:, 0] = np.arange(m + 1) * gap_penalty

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = f_matrix[i - 1, j - 1] + s_matrix[i - 1, j - 1]
            delete = f_matrix[i - 1, j] + gap_penalty
            insert = f_matrix[i, j - 1] + gap_penalty
            f_matrix[i, j] = max(match, delete, insert)

    aligned_indices = []
    i, j = m, n
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and f_matrix[i, j] == f_matrix[i - 1, j - 1] + s_matrix[i - 1, j - 1]
        ):
            aligned_indices.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and f_matrix[i, j] == f_matrix[i - 1, j] + gap_penalty:
            i -= 1
        else:
            j -= 1

    return np.array(aligned_indices[::-1])

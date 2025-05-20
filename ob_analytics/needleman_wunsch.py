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

    def similarity_score(t1: pd.Timestamp, t2: pd.Timestamp) -> float:
        """
        Calculate the similarity score between two timestamps.

        Parameters
        ----------
        t1 : pandas.Timestamp
            The first timestamp.
        t2 : pandas.Timestamp
            The second timestamp.

        Returns
        -------
        float
            The similarity score based on the time difference.
        """
        diff_ms = abs((t1 - t2).total_seconds() * 1000)
        return cut_off_ms / diff_ms if diff_ms != 0 else cut_off_ms

    similarity_matrix = np.zeros((len(a), len(b)), dtype="float64")
    for i, t1 in enumerate(a):
        for j, t2 in enumerate(b):
            similarity_matrix[i, j] = similarity_score(t1, t2)

    return similarity_matrix


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

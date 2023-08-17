import numpy as np
import pandas as pd

def vector_diff(v: list[float]) -> list[float]:
    """Calculate the difference between consecutive elements of a list.
    
    Parameters
    ----------
    v : list[float]
        Input list of numbers.

    Returns
    -------
    list[float]
        List containing the difference between consecutive elements.
    """
    return np.insert(np.diff(v), 0, 0).tolist()

def reverse_matrix(m: np.ndarray) -> np.ndarray:
    """Reverse the rows of a matrix.
    
    Parameters
    ----------
    m : np.ndarray
        Input matrix.

    Returns
    -------
    np.ndarray
        Matrix with reversed rows.
    """
    return m[::-1]

def norml(v: list[float], minv: float = None, maxv: float = None) -> list[float]:
    """Normalize a list of numbers to the range [0, 1].
    
    Parameters
    ----------
    v : list[float]
        Input list of numbers.
    minv : float, optional
        Minimum value to be used for normalization. If not specified, the minimum of the list is used.
    maxv : float, optional
        Maximum value to be used for normalization. If not specified, the maximum of the list is used.

    Returns
    -------
    list[float]
        Normalized list of numbers.
    """
    v_array = np.array(v)
    if minv is None:
        minv = v_array.min()
    if maxv is None:
        maxv = v_array.max()
    return ((v_array - minv) / (maxv - minv)).tolist()

def to_zoo(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Convert a DataFrame to a time-series DataFrame using the specified timestamp column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    timestamp_col : str
        Name of the column to be used as the timestamp. Default is "timestamp".

    Returns
    -------
    pd.DataFrame
        Time-series DataFrame indexed by the timestamp column.
    """
    return df.set_index(timestamp_col)

def interval_sum_breaks(v: list[float], breaks: list[int]) -> list[float]:
    """Compute the cumulative sum of a list at specified intervals.
    
    Parameters
    ----------
    v : list[float]
        Input list of numbers.
    breaks : list[int]
        Indices at which to compute the cumulative sums.

    Returns
    -------
    list[float]
        List of cumulative sums at the specified intervals.
    """
    cs = np.cumsum(v)
    intervals = cs[breaks]
    return np.insert(np.diff(intervals, prepend=0), 0, intervals[0]).tolist()

def vwap(price: list[float], volume: list[float]) -> float:
    """Compute the Volume Weighted Average Price (VWAP).
    
    Parameters
    ----------
    price : list[float]
        List of prices.
    volume : list[float]
        Corresponding list of volumes.

    Returns
    -------
    float
        Volume Weighted Average Price.
    """
    return np.dot(price, volume) / np.sum(volume)

def interval_vwap(price: list[float], volume: list[float], breaks: list[int]) -> list[float]:
    """Compute the VWAP at specified intervals.
    
    Parameters
    ----------
    price : list[float]
        List of prices.
    volume : list[float]
        Corresponding list of volumes.
    breaks : list[int]
        Indices at which to compute the VWAP.

    Returns
    -------
    list[float]
        List of VWAP values at the specified intervals.
    """
    vwap_values = []
    start_idx = 0
    for end_idx in breaks:
        interval_price = price[start_idx:end_idx+1]
        interval_volume = volume[start_idx:end_idx+1]
        vwap_values.append(vwap(interval_price, interval_volume))
        start_idx = end_idx + 1
    return vwap_values

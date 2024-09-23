## auxillary.py

import numpy as np
import pandas as pd


def vector_diff(v: np.ndarray) -> np.ndarray:
    """
    Calculate the difference between consecutive elements of a vector.

    Args:
      v: A NumPy array of numerical values.

    Returns:
      A NumPy array with the same length as v, where the first element is 0
      and the remaining elements are the differences between consecutive elements
      of v.
    """
    return np.diff(v, prepend=0)


def reverse_matrix(m):
    """
    Reverse the rows of a DataFrame or 2D NumPy array.

    Args:
        m: A pandas DataFrame or 2D NumPy array.

    Returns:
        A DataFrame or 2D NumPy array with the rows reversed.
    """
    # For a pandas DataFrame
    if isinstance(m, pd.DataFrame):
        return m.iloc[::-1].reset_index(drop=True)
    # For a NumPy array
    elif isinstance(m, np.ndarray):
        return m[::-1, :]
    else:
        raise TypeError("Input must be a pandas DataFrame or a NumPy array.")

def norml(v: np.ndarray, minv: float = None, maxv: float = None) -> np.ndarray:
    """
    Normalize a vector to the range [0, 1].

    Args:
      v: A NumPy array of numerical values.
      minv: The minimum value for normalization. If None, the minimum of v is used.
      maxv: The maximum value for normalization. If None, the maximum of v is used.

    Returns:
      A NumPy array with the same shape as v, where each element is normalized
      to the range [0, 1].
    """
    minv = np.min(v) if minv is None else minv
    maxv = np.max(v) if maxv is None else maxv
    return (v - minv) / (maxv - minv)


def to_pandas(v: np.ndarray) -> pd.DataFrame:
    """
    Convert a NumPy array to a pandas DataFrame.

    Args:
      v: A 2D NumPy array, where the first column is assumed to be timestamps.

    Returns:
      A pandas DataFrame representing the time series data.
    """
    df = pd.DataFrame(v[:, 1:], columns=[f"col{i}" for i in range(1, v.shape[1])])
    df["timestamp"] = pd.to_datetime(v[:, 0])
    df.set_index("timestamp", inplace=True)
    return df


def interval_sum_breaks(v: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """
    Calculate the sum of values in each interval defined by breaks.

    Args:
      v: A NumPy array of numerical values.
      breaks: A NumPy array of indices that define the intervals.

    Returns:
      A NumPy array with the sum of values in each interval.
    """
    cs = np.cumsum(v)
    intervals = cs[breaks]
    return np.concatenate(([intervals[0]], np.diff(intervals)))


def vwap(price: np.ndarray, volume: np.ndarray) -> float:
    """
    Calculate the volume-weighted average price (VWAP).

    Args:
      price: A NumPy array of prices.
      volume: A NumPy array of volumes.

    Returns:
      The VWAP as a float.
    """
    return np.average(price, weights=volume)


def interval_vwap(
    price: np.ndarray, volume: np.ndarray, breaks: np.ndarray
) -> np.ndarray:
    """
    Calculate the VWAP for each interval defined by breaks.

    Args:
      price: A NumPy array of prices.
      volume: A NumPy array of volumes.
      breaks: A NumPy array of indices that define the intervals.

    Returns:
      A NumPy array with the VWAP for each interval.
    """
    return interval_sum_breaks(price * volume, breaks) / interval_sum_breaks(
        volume, breaks
    )


def interval_price_level_gaps(volume: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """
    Calculate the number of price level gaps in each interval.

    Args:
      volume: A NumPy array of volumes.
      breaks: A NumPy array of indices that define the intervals.

    Returns:
      A NumPy array with the number of price level gaps in each interval.
    """
    return interval_sum_breaks(np.where(volume == 0, 1, 0), breaks)

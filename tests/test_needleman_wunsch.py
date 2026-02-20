"""Tests for ob_analytics._needleman_wunsch."""


import numpy as np
import pandas as pd

from ob_analytics._needleman_wunsch import align_sequences, create_similarity_matrix


class TestCreateSimilarityMatrix:
    def test_identical_timestamps(self):
        ts = pd.Timestamp("2015-01-01 00:00:00")
        a = pd.Series([ts, ts + pd.Timedelta(seconds=1)])
        b = pd.Series([ts, ts + pd.Timedelta(seconds=1)])
        result = create_similarity_matrix(a, b, cut_off_ms=1000)
        assert result.shape == (2, 2)
        assert result[0, 0] == 1000  # diff=0 → cut_off_ms
        assert result[1, 1] == 1000

    def test_asymmetric_shape(self):
        ts = pd.Timestamp("2015-01-01")
        a = pd.Series([ts, ts + pd.Timedelta(seconds=1), ts + pd.Timedelta(seconds=2)])
        b = pd.Series([ts + pd.Timedelta(milliseconds=500)])
        result = create_similarity_matrix(a, b, cut_off_ms=5000)
        assert result.shape == (3, 1)

    def test_score_inversely_proportional_to_distance(self):
        ts = pd.Timestamp("2015-01-01")
        a = pd.Series([ts])
        b = pd.Series([ts + pd.Timedelta(seconds=1), ts + pd.Timedelta(seconds=2)])
        result = create_similarity_matrix(a, b, cut_off_ms=5000)
        assert result[0, 0] > result[0, 1]

    def test_zero_diff_returns_cutoff(self):
        ts = pd.Timestamp("2015-01-01")
        a = pd.Series([ts])
        b = pd.Series([ts])
        result = create_similarity_matrix(a, b, cut_off_ms=3000)
        assert result[0, 0] == 3000


class TestAlignSequences:
    def test_perfect_diagonal(self):
        s_matrix = np.array([[10.0, 0.0], [0.0, 10.0]])
        result = align_sequences(s_matrix)
        np.testing.assert_array_equal(result, [[0, 0], [1, 1]])

    def test_single_element(self):
        s_matrix = np.array([[5.0]])
        result = align_sequences(s_matrix)
        np.testing.assert_array_equal(result, [[0, 0]])

    def test_more_rows_than_cols(self):
        s_matrix = np.array([[10.0], [5.0], [1.0]])
        result = align_sequences(s_matrix)
        assert result.shape[1] == 2
        assert 0 in result[:, 1]

    def test_more_cols_than_rows(self):
        s_matrix = np.array([[10.0, 1.0, 0.5]])
        result = align_sequences(s_matrix)
        assert 0 in result[:, 0]

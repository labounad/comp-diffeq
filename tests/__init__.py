# tests/test_fem1d.py

import numpy as np
from fem.fem1d import _elem_indices


def test_elem_indices_basic():
    result = _elem_indices(3, 1)
    expected = np.array([
        [0, 1],
        [1, 2],
        [2, 3]
    ])
    np.testing.assert_array_equal(result, expected)


def test_elem_indices_general():
    result = _elem_indices(2, 2)
    expected = np.array([
        [0, 1, 2],
        [2, 3, 4]
    ])
    np.testing.assert_array_equal(result, expected)

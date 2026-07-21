import pytest
import numpy as np
from assg_tasks import softmax


def test_matrix():
    x = np.array(
        [[9, 2, 5, 0, 0],
            [7, 5, 0, 0 ,0]]
    )
    expected_s = np.array(
        [[9.80897665e-01, 8.94462891e-04, 1.79657674e-02, 1.21052389e-04, 1.21052389e-04],
            [8.78679856e-01, 1.18916387e-01, 8.01252314e-04, 8.01252314e-04, 8.01252314e-04]]
    )
    s = softmax(x)

    assert np.allclose(s, expected_s)
    assert s.shape == (2,5)

    # the rows of softmax are a normalized probability space so should sum up to 0
    assert np.allclose(np.sum(s, axis=1), np.array([1.0, 1.0]))

    # check that original x was not modified
    assert not np.allclose(s, x)


def test_random_matrix():
    x = np.random.random((20,10))
    s = softmax(x)

    assert s.shape == (20,10)

    # the rows of softmax are a normalized probability space so should sum up to 0
    assert np.allclose(np.sum(s, axis=1), np.ones((20,)))

    # check that original x was not modified
    assert not np.allclose(s, x)

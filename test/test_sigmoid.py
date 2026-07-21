import pytest
import numpy as np
from assg_tasks import sigmoid


def test_input_scalar():
    assert sigmoid(3) == pytest.approx(0.9525741268224334)

def test_input_vector():
    # test a 1-d vector
    x = np.array([-5, 0, 3])
    s = sigmoid(x)
    assert np.allclose(s, np.array([0.00669285, 0.5, 0.95257413]))

def test_input_matrix():
    # test a 3-d tensor
    x = np.linspace(-5, 5, 27).reshape((3,3,3))
    s = sigmoid(x)
    expected_s = np.array(
        [[[0.00669285, 0.00980136, 0.01433278],
            [0.02091496, 0.03042661, 0.04406926],
            [0.06342879, 0.09048789, 0.12751884]],
        
            [[0.17675903, 0.23978727, 0.31664553],
            [0.40501421, 0.5,        0.59498579],
            [0.68335447, 0.76021273, 0.82324097]],
        
            [[0.87248116, 0.90951211, 0.93657121],
            [0.95593074, 0.96957339, 0.97908504],
            [0.98566722, 0.99019864, 0.99330715]]]
    )
    assert np.allclose(s, expected_s)

def test_input_list():
    # a regular list still does not work for a vectorized function
    with pytest.raises(Exception) as excinfo:
        x = [-5, 0, 3]
        s = sigmoid(x)
    assert "TypeError" in str(excinfo)
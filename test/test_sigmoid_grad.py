import pytest
import numpy as np
from assg_tasks import sigmoid_grad


def test_input_scalar():
    s = sigmoid_grad(3)
    assert s == pytest.approx(0.045176659730912)

def test_input_vector():
    # test a 1-d vector
    x = np.array([-5, 0, 3])
    s = sigmoid_grad(x)
    assert np.allclose(s, np.array([0.00664806, 0.25, 0.04517666]))

def test_input_matrix():
    # test a 3-d tensor
    x = np.linspace(-5, 5, 27).reshape((3,3,3))
    s = sigmoid_grad(x)
    expected_s = np.array(
    [[[0.00664806, 0.00970529, 0.01412736],
        [0.02047752, 0.02950084, 0.04212716],
        [0.05940558, 0.08229983, 0.11125779]],
    
        [[0.14551528, 0.18228933, 0.21638114],
        [0.2409777,  0.25,       0.2409777 ],
        [0.21638114, 0.18228933, 0.14551528]],
    
        [[0.11125779, 0.08229983, 0.05940558],
        [0.04212716, 0.02950084, 0.02047752],
        [0.01412736, 0.00970529, 0.00664806]]]
    )
    assert np.allclose(s, expected_s)

def test_input_list():
    # a regular list still does not work for a vectorized function
    with pytest.raises(Exception) as excinfo:
        x = [-5, 0, 3]
        s = sigmoid_grad(x)
    assert "TypeError" in str(excinfo)
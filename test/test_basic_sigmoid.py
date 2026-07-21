import pytest
import numpy as np
from assg_tasks import basic_sigmoid


def test_input_3():
    assert basic_sigmoid(3) == pytest.approx(0.9525741268224334)

def test_input_0():
    assert basic_sigmoid(0) == pytest.approx(0.5)
    
def test_input_neg5():
    assert basic_sigmoid(-5) == pytest.approx(0.0066928509242848554)

def test_using_math_library():
    # if using math library should not be vectorized, expect an exception
    # if we pass a numpy array
    with pytest.raises(Exception) as excinfo:
        x = np.array([1, 2, 3])
        basic_sigmoid(x)
    assert "only 0-dimensional arrays can be converted to Python scalars" in str(excinfo.value)
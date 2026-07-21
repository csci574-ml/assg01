import pytest
import random
import numpy as np
from assg_tasks import mae


def test_case1():
    y_pred = np.array([.9, 0.2, 0.1, .4, .9])
    y_true = np.array([1, 0, 0, 1, 1])
    loss = mae(y_pred, y_true)
    
    assert loss == pytest.approx(0.22000000000000003)


def test_case2():
    y_pred = np.array([0.8, 0.3, 0.7, 0.5, 0.8, 0.6, 0.5, 0.2])
    y_true = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    loss = mae(y_pred, y_true)

    assert loss == pytest.approx(0.35)

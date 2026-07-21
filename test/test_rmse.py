import pytest
import numpy as np
from assg_tasks import rmse

def test_case1():
    y_pred = np.array([.9, 0.2, 0.1, .4, .9])
    y_true = np.array([1, 0, 0, 1, 1])
    loss = rmse(y_pred, y_true)

    assert loss == pytest.approx(0.29325756597230357)


def test_case2():
    y_pred = np.array([0.8, 0.3, 0.7, 0.5, 0.8, 0.6, 0.5, 0.2])
    y_true = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    loss = rmse(y_pred, y_true)

    assert loss == pytest.approx(0.3807886552931954)
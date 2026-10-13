"""Tests for the implemented functions."""

import numpy as np
from implementations import (
    grad_err,
    grad_logistic,
    logistic_loss,
    mean_squared_error_gd,
    mean_squared_error_sgd,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
)
from numpy.testing import assert_allclose

''' Toy datasets '''
# For regression: A perfect line y = 1 + 2x

x_reg = np.array([0., 1., 2., 3.])
tx_reg = np.column_stack([np.ones(len(x_reg)), x_reg])

y_reg = np.array([1., 3., 5., 7.])

w_reg_0 = np.zeros(2)
expected_w = [1., 2.]

# For logistic regression

tx_log = np.array([
    [1., -2.],
    [1., -1.],
    [1.,  1.],
    [1.,  2.],
])

y_log = np.array([0., 0., 1., 1.])

w_log_0 = np.zeros(2)

'''Tests'''
def test_grad_err():
    gradient, err = grad_err(y_reg, tx_reg, w_reg_0)

    expected_gradient = np.array([-4., -8.5])
    expected_err = np.array([1., 3., 5., 7.])

    assert_allclose(gradient, expected_gradient)
    assert_allclose(err, expected_err)

    
def test_least_squares_exact_sol():
    w, loss = least_squares(y_reg, tx_reg)

    assert_allclose(w, expected_w, atol=1e-10)
    assert np.isclose(loss, 0.0)


def test_mse_gd_one_step():
    w, loss = mean_squared_error_gd(y_reg, tx_reg, w_reg_0.copy(), max_iters=1, gamma=0.05)
    expected_w_after_one_step = np.array([0.2, 0.425])

    assert_allclose(w, expected_w_after_one_step, atol=1e-10)
                                    
def test_mse_gd_converges_to_least_squares():
    w_expected, _ = least_squares(y_reg, tx_reg)
    w, loss = mean_squared_error_gd(y_reg, tx_reg, w_reg_0.copy(), max_iters=5000, gamma=0.05)

    assert_allclose(w, w_expected, atol=1e-2)
    assert loss < 1e-3


def test_mse_sgd_approximately_converges():
    np.random.seed(42)

    w_expected, _ = least_squares(y_reg, tx_reg)
    w, loss = mean_squared_error_sgd(y_reg, tx_reg, w_reg_0.copy(), max_iters=10000, gamma=0.01)

    assert_allclose(w, w_expected, atol=0.2)
    assert np.isfinite(loss)


def test_ridge_lambda_zero_equals_least_squares():
    w_ls, _ = least_squares(y_reg, tx_reg)

    w_ridge, _ = ridge_regression(y_reg, tx_reg, lambda_=0)
    assert_allclose(w_ridge, w_ls, atol=1e-10)


def test_logistic_loss_at_zero_weights():
    loss = logistic_loss(y_log, tx_log, w_log_0)

    assert np.isclose(loss, np.log(2), atol=1e-10)


def test_grad_logistic_at_zero_weights():
    gradient = grad_logistic(y_log, tx_log, w_log_0)
    expected = np.array([0., -0.75])

    assert_allclose(gradient, expected, atol=1e-10)

def test_logistic_regression_learns_simple_problem():
    w, loss = logistic_regression(y_log, tx_log, w_log_0, max_iters=1000, gamma=0.1)

    scores = tx_log @ w
    probabilities = 1 / (1 + np.exp(-scores))

    assert probabilities[0] < 0.5
    assert probabilities[1] < 0.5
    assert probabilities[2] > 0.5
    assert probabilities[3] > 0.5

    assert np.isfinite(loss)


def test_reg_logistic_lambda_zero_equals_logistic():
    w1, loss1 = logistic_regression(y_log, tx_log, w_log_0, max_iters=100, gamma=0.1)
    w2, loss2 = reg_logistic_regression(y_log, tx_log, lambda_=0, initial_w=w_log_0, max_iters=100, gamma=0.1)

    assert_allclose(w1, w2)
    assert_allclose(loss1, loss2)
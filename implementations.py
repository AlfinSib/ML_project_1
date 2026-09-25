import numpy as np

def grad_err(y, tx, w):
    err = y - tx.dot(w)
    if np.ndim(err) == 0:
        grad = -tx.T.dot(err)
    else:
        grad = -tx.T.dot(err) / len(err)
    return grad, err

def mse(err):
    return 0.5 * np.mean(err ** 2)

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for _ in range(max_iters):
        grad, err = grad_err(y, tx, w)
        w = w - gamma * grad
    _, err = grad_err(y, tx, w)
    return (w, mse(err))

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for _ in range(max_iters):
        idx = np.random.choice(len(y))
        grad, err = grad_err(y[idx], tx[idx,:], w)
        w = w - gamma * grad
    _, err = grad_err(y, tx, w)
    return (w, mse(err))

def least_squares(y, tx):
    w = np.linalg.solve(np.dot(tx.T, tx), tx.T.dot(y))
    return (w, mse(y - tx.dot(w)))

def ridge_regression(y, tx, lambda_):
    n, d = np.shape(tx)
    lhs = np.dot(tx.T, tx) + 2* n * lambda_ * np.identity(d)
    w = np.linalg.solve(lhs, tx.T.dot(y))
    return (w, mse(y - tx.dot(w)))


# Logistic Regression part not done as they are not yet covered by lectures  

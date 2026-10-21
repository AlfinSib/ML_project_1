import numpy as np

def grad_err(y, tx, w):
    err = y - tx.dot(w)
    if np.ndim(err) == 0:
        grad = -tx.T.dot(err)
    else:
        grad = -tx.T.dot(err) / len(err)
    return grad, err

def grad_logistic(y, tx, w):
    sigmoid_tx = 0.5 * (1 + np.tanh(0.5 * tx.dot(w))) # An experiment to avoid overflow problem
    if np.ndim(y) == 0:
        return tx.T.dot(sigmoid_tx - y)
    else:
        return tx.T.dot(sigmoid_tx - y) / len(y)

def mse(err):
    return 0.5 * np.mean(err ** 2)

def logistic_loss(y, tx, w):
    sigmoid_tx = 0.5 * (1 + np.tanh(0.5 * tx.dot(w))) # An experiment to avoid overflow problem
    loss = - np.mean(y * np.log(sigmoid_tx) + (1 - y) * np.log(1 - sigmoid_tx))
    return loss

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

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for _ in range(max_iters):
        grad = grad_logistic(y, tx, w)
        w = w - gamma * grad
    loss = logistic_loss(y, tx, w)
    return (w, loss)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    for _ in range(max_iters):
        grad = grad_logistic(y, tx, w)
        w = w - gamma * (grad + 2 * lambda_ * w)
    loss = logistic_loss(y, tx, w)
    return (w, loss)    

def ROC_AUC(y_true, y_score):
    order = np.argsort(-y_score)                       # highest scores first
    s_sorted = y_score[order]
    y_sorted = (y_true[order] == 1)
    keep = np.r_[np.diff(s_sorted) != 0, True]       
    true_pos = np.cumsum(y_sorted)[keep]
    false_pos = np.cumsum(~y_sorted)[keep]
    n_pos, n_neg = y_sorted.sum(), (~y_sorted).sum()
    true_pos_rate = np.r_[0, true_pos / n_pos]
    false_pos_rate = np.r_[0, false_pos / n_neg]
    return false_pos_rate, true_pos_rate, np.trapezoid(true_pos_rate, false_pos_rate)

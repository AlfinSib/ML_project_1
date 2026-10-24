import numpy as np

def ROC_AUC(y_true, y_score):    
    """
    Computes the ROC and AUC of a model using true outcome and predicted score (i.e. before being made binary)
    Args:
        y_true: true outcome
        y_score: the score (probability) computed by the model
    Returns:
        false positive rate, true positive rate and AUC
    """
    order = np.argsort(-y_score)                       
    s_sorted = y_score[order]
    y_sorted = (y_true[order] == 1)
    keep = np.r_[np.diff(s_sorted) != 0, True]       
    true_pos = np.cumsum(y_sorted)[keep]
    false_pos = np.cumsum(~y_sorted)[keep]
    n_pos, n_neg = y_sorted.sum(), (~y_sorted).sum()
    true_pos_rate = np.r_[0, true_pos / n_pos]
    false_pos_rate = np.r_[0, false_pos / n_neg]
    return false_pos_rate, true_pos_rate, np.trapezoid(true_pos_rate, false_pos_rate)

def F1_score(y_true, y_score, p):
    """
    Compute the F1 score
    Args:
        y_true: true labels {0, 1}
        y_score: predicted score, a probability
        p: the threshold p s.t. if y_score>p, then our predicted y will be 1
    Returns:
        f1 (float). Returns 0.0 if nothing is predicted positive.
    """
    y_pred = y_score > p
    y_true = (y_true == 1)
    true_pos = np.sum(y_pred & y_true)
    false_pos = np.sum(y_pred & ~y_true)
    false_neg = np.sum(~y_pred & y_true)
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg)
    if 2 * true_pos + false_pos + false_neg == 0:                    # nothing predicted positive anywhere
        return 0.0, (0, 0, 0)
    else:
        f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg)
        return f1, (int(true_pos), int(false_pos), int(false_neg))
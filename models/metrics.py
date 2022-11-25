from typing import List,  Dict

from sklearn.metrics import accuracy_score, recall_score, precision_score

from .utils import binarize_labels

def pure_accuracy(y_true:List[int], y_pred:List[int]) -> float:
    return accuracy_score(y_true, y_pred)

def penalize_out_scope_errors(y_true:List[int], y_pred:List[int]) -> float:
    out_scope_label = "out_of_scope"
    y_true_bin = binarize_labels(y_true, out_scope_label)
    y_pred_bin = binarize_labels(y_pred, out_scope_label)
    return recall_score(y_true_bin, y_pred_bin)

def penalize_luggage_lost_errors(y_true:List[int], y_pred:List[int]) -> float:
    lost_luggage_label = "lost_luggage"
    y_true_bin = binarize_labels(y_true, lost_luggage_label)
    y_pred_bin = binarize_labels(y_pred, lost_luggage_label)
    return precision_score(y_true_bin, y_pred_bin)
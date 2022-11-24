from sklearn.metrics import accuracy_score, precision_score, recall_score

from vectorization import binarize_labels

def pure_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def penalize_out_scope_errors(y_true, y_pred, label_dict):
    out_scope_label = label_dict["out_of_scope"]
    y_true_bin = binarize_labels(y_true, out_scope_label)
    y_pred_bin = binarize_labels(y_pred, out_scope_label)
    return recall_score(y_true_bin, y_pred_bin)

def penalize_luggage_lost_errors(y_true, y_pred, label_dict):
    lost_luggage_label = label_dict["lost_luggage"]
    y_true_bin = binarize_labels(y_true, lost_luggage_label)
    y_pred_bin = binarize_labels(y_pred, lost_luggage_label)
    return precision_score(y_true_bin, y_pred_bin)
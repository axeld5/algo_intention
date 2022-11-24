import numpy as np

from typing import List, Dict

from .metrics import pure_accuracy, penalize_luggage_lost_errors, penalize_out_scope_errors

class MLModel:
    
    def __init__(self, model) -> None:
        self.model = model

    def fit(self, vect_texts:np.array, vect_labels:List[int]) -> None:
        self.model.fit(vect_texts, vect_labels)

    def predict(self, vect_texts:np.array) -> List[int]:
        return self.model.predict(vect_texts)

    def evaluate(self, vect_texts:np.array, vect_labels:List[int], label_dict:Dict[str, int]) -> Dict[str, float]:
        metric_dict = {}
        pred_labels = self.predict(vect_texts)
        metric_dict["accuracy"] = pure_accuracy(vect_labels, pred_labels)
        metric_dict["lost_lug_pen"] = penalize_luggage_lost_errors(vect_labels, pred_labels, label_dict)
        metric_dict["out_scope_err"] = penalize_out_scope_errors(vect_labels, pred_labels, label_dict)
        return metric_dict
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Tuple
from .metrics import pure_accuracy, penalize_luggage_lost_errors, penalize_out_scope_errors

class MLModel:
    
    def __init__(self, model) -> None:
        self.model = model
        self.tokenizer = TfidfVectorizer()
        self.encoder = LabelEncoder()

    def fit(self, texts:List[str], labels:List[str]) -> None:
        vect_texts, vect_labels = self.train_tokenizer_(texts, labels)
        self.model.fit(vect_texts, vect_labels)

    def predict(self, texts:List[str]) -> List[str]:
        vect_texts = self.tokenizer.transform(texts)
        predicted_labels = self.encoder.inverse_transform(self.model.predict(vect_texts)).tolist()
        return predicted_labels
    
    def train_tokenizer_(self, texts:List[str], labels:List[str]) -> Tuple[np.ndarray, List[int]]:
        vect_texts = self.tokenizer.fit_transform(texts)
        vect_labels = self.encode_labels_(labels)
        return vect_texts, vect_labels
    
    def encode_labels_(self, labels:List[str]) -> np.ndarray:
        vect_labels = self.encoder.fit_transform(labels)
        return vect_labels


    def evaluate_metrics(self, texts:List[str], labels:List[str]) -> Dict[str, float]:
        metric_dict = {}
        pred_labels = self.predict(texts)
        metric_dict["accuracy"] = pure_accuracy(labels, pred_labels)
        metric_dict["lost_lug_pen"] = penalize_luggage_lost_errors(labels, pred_labels)
        metric_dict["out_scope_err"] = penalize_out_scope_errors(labels, pred_labels)
        return metric_dict
import pandas as pd  
import numpy as np
import torch

from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD

def vectorize_data(texts:List[str], labels:List[str]) -> Tuple[np.array, np.array, Dict[str, int]]:
    vectorizer = TfidfVectorizer()
    vect_texts = vectorizer.fit_transform(texts)
    vect_labels, label_dict = encode_labels(labels)
    return vect_texts, vect_labels, label_dict

def reduce_data(vect_texts:np.array) -> np.array:
    reducer = TruncatedSVD(n_components=2)
    data_reduced = reducer.fit_transform(vect_texts)
    return data_reduced

def encode_labels(labels:List[str]) -> Tuple[np.array, Dict[int, str]]:
    encoder = LabelEncoder()
    vect_labels = encoder.fit_transform(labels)
    label_dict = {}
    ordered_labels = encoder.classes_ 
    for i in range(9):
        label = ordered_labels[i]
        label_dict[i] = label
    return vect_labels, label_dict
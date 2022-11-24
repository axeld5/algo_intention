import pandas as pd  
import numpy as np
import torch

from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD

def vectorize_data(file_name:str) -> Tuple[np.array, np.array, Dict[str, int]]:
    data = pd.read_csv(file_name)
    texts = data['text'].tolist()
    labels = data['label'].tolist()
    vectorizer = TfidfVectorizer()
    vect_texts = vectorizer.fit_transform(texts)
    vect_labels, label_dict = encode_labels(labels)
    return vect_texts, vect_labels, label_dict

def reduce_data(vect_texts:np.array) -> np.array:
    reducer = TruncatedSVD(n_components=2)
    data_reduced = reducer.fit_transform(vect_texts)
    return data_reduced

def encode_labels(labels:List[str]) -> Tuple[np.array, Dict[str, int]]:
    encoder = LabelEncoder()
    vect_labels = encoder.fit_transform(labels)
    label_dict = {}
    ordered_labels = encoder.classes_ 
    for i in range(9):
        label = ordered_labels[i]
        label_dict[label] = i
    return vect_labels, label_dict

class ToTorchDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def binarize_labels(labels:List[int], label:int) -> List[int]:
    encoded_labels = [0]*len(labels)
    for i in range(len(labels)):
        if labels[i] == label:
            encoded_labels[i] = 1
    return encoded_labels 
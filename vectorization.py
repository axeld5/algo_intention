import pandas as pd  
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.decomposition import TruncatedSVD

def vectorize_data(file_name):
    data = pd.read_csv(file_name)
    texts = data['text'].tolist()
    labels = data['label'].tolist()
    vectorizer = TfidfVectorizer()
    vect_texts = vectorizer.fit_transform(texts)
    binary_labels = binarize_labels(labels, "out_of_scope")
    vect_labels, label_dict = encode_labels(labels)
    return vect_texts, binary_labels, vect_labels, label_dict


def reduce_data(vect_texts):
    reducer = TruncatedSVD(n_components=2)
    data_reduced = reducer.fit_transform(vect_texts)
    return data_reduced

def binarize_labels(labels, label):
    encoded_labels = [0]*len(labels)
    for i in range(len(labels)):
        if labels[i] == label:
            encoded_labels[i] = 1
    return encoded_labels 

def encode_labels(labels):
    encoder = LabelEncoder()
    vect_labels = encoder.fit_transform(labels)
    label_dict = {}
    ordered_labels = encoder.classes_ 
    for i in range(9):
        label = ordered_labels[i]
        label_dict[label] = i
    return vect_labels, label_dict

def oh_encode_labels(labels):
    encoder = OneHotEncoder(sparse=False)
    labels = (np.array(labels)).reshape(-1,1)
    vect_labels = encoder.fit_transform(labels)
    return vect_labels
import pandas as pd  

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD

def vectorize_data(file_name):
    data = pd.read_csv(file_name)
    texts = data['text'].tolist()
    labels = data['label'].tolist()
    vectorizer = TfidfVectorizer()
    vect_texts = vectorizer.fit_transform(texts)
    binary_labels = binarize_labels(labels)
    vect_labels = encode_labels(labels)
    return vect_texts, binary_labels, vect_labels


def reduce_data(vect_texts):
    reducer = TruncatedSVD(n_components=2)
    data_reduced = reducer.fit_transform(vect_texts)
    return data_reduced

def binarize_labels(labels):
    encoded_labels = [0]*len(labels)
    for i in range(len(labels)):
        if labels[i] == "out_of_scope":
            encoded_labels[i] = 1
    return encoded_labels 

def encode_labels(labels):
    encoder = LabelEncoder()
    vect_labels = encoder.fit_transform(labels)
    return vect_labels
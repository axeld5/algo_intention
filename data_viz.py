import argparse
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns

from utils import vectorize_data, reduce_data
from models.utils import binarize_labels

def show_label_occ(filename:str) -> None:    
    data = pd.read_csv(filename)
    labels = data['label'].tolist()
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_dict = dict(zip(unique_labels, counts))
    label_occurences = [label_dict[val] for val in unique_labels]

    plt.figure(figsize=(15,10))
    df = pd.DataFrame({'labels' : unique_labels, 'label_occurences': label_occurences})
    sns.barplot(data=df, x="labels", y="label_occurences")
    plt.title("Label occurences in the data")
    plt.show()

def plot_classes(filename:str) -> None:    
    data = pd.read_csv(filename)    
    texts = data['text'].tolist()
    labels = data['label'].tolist()
    vect_texts, vect_labels, label_dict = vectorize_data(texts, labels)
    reduced_texts = reduce_data(vect_texts)
    binary_labels = binarize_labels(labels, "out_of_scope")
    fig, ax = plt.subplots(1, 2, figsize=(15,10))
    bin_scatter = ax[0].scatter(reduced_texts[:,0], reduced_texts[:,1], c=binary_labels)
    ax[0].set_title("plot of data in_scope vs out_of_scope")
    bin_scat_elem = bin_scatter.legend_elements()
    bin_scat_elem[1][0] = "in_scope"
    bin_scat_elem[1][1] = "out_of_scope"
    ax[0].legend(*bin_scat_elem,
                    loc="lower left", title="Classes")
    scatter = ax[1].scatter(reduced_texts[:,0], reduced_texts[:,1], c=vect_labels)
    ax[1].set_title("plot of labelled data")
    scat_elem = scatter.legend_elements()
    for i in range(len(scat_elem[1])):
        scat_elem[1][i] = label_dict[i]
    ax[1].legend(*scat_elem,
                    loc="lower left", title="Classes")
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("--filename")

args = parser.parse_args()

if __name__ == "__main__":
    show_label_occ(args.filename)
    plot_classes(args.filename)

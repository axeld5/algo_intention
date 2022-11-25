import matplotlib.pyplot as plt 
import numpy as np 

from models.utils import vectorize_data

if __name__ == "__main__":
    vect_texts, vect_labels, label_dict = vectorize_data("data/augmented-intent-detection-train.csv")
    info_dict = {}
    for label in label_dict.keys():
        info_dict[label] = np.count_nonzero(vect_labels == label_dict[label])
    print(info_dict)
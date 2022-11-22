import matplotlib.pyplot as plt 

from vectorization import vectorize_data, reduce_data

if __name__ == "__main__":
    vect_texts, binary_labels, vect_labels = vectorize_data("data/intent-detection-train.csv")
    reduced_texts = reduce_data(vect_texts)
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].scatter(reduced_texts[:,0], reduced_texts[:,1], c=binary_labels)
    ax[0].set_title("plot of data in_scope vs out_of_scope")
    ax[1].scatter(reduced_texts[:,0], reduced_texts[:,1], c=vect_labels)
    ax[1].set_title("plot of labelled data")
    plt.show()
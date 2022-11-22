from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from vectorization import vectorize_data


if __name__ == "__main__":
    vect_texts, bin_labels, vect_labels = vectorize_data("data/intent-detection-train.csv")
    train_texts, test_texts, train_labels, test_labels = train_test_split(vect_texts, bin_labels, test_size=0.2)
    clf_list = [RandomForestClassifier(), SVC(), AdaBoostClassifier()]
    for clf in clf_list: 
        clf.fit(train_texts, train_labels)
        pred_labels = clf.predict(test_texts)
        print(accuracy_score(test_labels, pred_labels))
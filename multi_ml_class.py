from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from metrics import pure_accuracy, penalize_luggage_lost_errors, penalize_out_scope_errors
from vectorization import vectorize_data


if __name__ == "__main__":
    vect_texts, bin_labels, vect_labels, label_dict = vectorize_data("data/intent-detection-train.csv")
    train_texts, test_texts, train_labels, test_labels = train_test_split(vect_texts, vect_labels, test_size=0.33)
    clf_list = [RandomForestClassifier()]
    for clf in clf_list: 
        clf.fit(train_texts, train_labels)
        pred_labels = clf.predict(test_texts)
        print(pure_accuracy(test_labels, pred_labels))
        print(penalize_out_scope_errors(test_labels, pred_labels, label_dict))
        print(penalize_luggage_lost_errors(test_labels, pred_labels, label_dict))
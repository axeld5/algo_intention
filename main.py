import argparse 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from utils import vectorize_data
from models.ml_model import MLModel

parser = argparse.ArgumentParser()
parser.add_argument("--filename")

args = parser.parse_args()

if __name__ == "__main__":
    vect_texts, vect_labels, label_dict = vectorize_data(args.filename)
    train_texts, test_texts, train_labels, test_labels = train_test_split(vect_texts, vect_labels, test_size=0.33)
    model = MLModel(RandomForestClassifier())
    model.fit(train_texts, train_labels) 
    print(model.evaluate(test_texts, test_labels, label_dict))
import argparse 
import pandas as pd 

from joblib import load

parser = argparse.ArgumentParser()
parser.add_argument("--filename")

args = parser.parse_args()

if __name__ == "__main__":
    data = pd.read_csv(args.filename)
    texts = data['text'].tolist()
    bert_model = load("./saved_models/bert_model.joblib")
    predicted_bert_labels = bert_model.predict(texts)
    print(predicted_bert_labels) 
    rf_model = load("./saved_models/random_forest.joblib")
    predicted_rf_labels = rf_model.predict(texts)
    print(predicted_rf_labels)
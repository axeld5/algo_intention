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
    data["label"] = predicted_bert_labels
    data_adress = (args.filename).split('.')[0]
    data.to_csv(data_adress+"-inferred.csv", index=False)
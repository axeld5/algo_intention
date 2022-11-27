import time 

from joblib import load

if __name__ == "__main__":
    verbatim = ["Bonjour, quelle est la racine carree de pi"]
    model_dict = {"bert_model":0, "log_regression":0, "random_forest":0}
    model_saved = {}
    for model_name in model_dict.keys():
        model_saved[model_name] = load("./saved_models/" + model_name + ".joblib")
    for model_name in model_dict.keys():
        start = time.time()
        model = model_saved[model_name]
        model.predict(verbatim)
        model_dict[model_name] = time.time() - start
    print(model_dict)
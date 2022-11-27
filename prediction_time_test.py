import time 
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns

from joblib import load

if __name__ == "__main__":
    verbatim = ["Bonjour, quelle est la racine carree de pi"]
    time_dict = {"bert_model":0, "log_regression":0, "random_forest":0}
    model_saved = {}
    for model_name in time_dict.keys():
        model_saved[model_name] = load("./saved_models/" + model_name + ".joblib")
    for model_name in time_dict.keys():
        start = time.time()
        model = model_saved[model_name]
        model.predict(verbatim)
        time_dict[model_name] = time.time() - start
    time_df = pd.DataFrame({"model_names":list(time_dict.keys()), "time_values":list(time_dict.values())})
    sns.barplot(data=time_df, x="model_names", y="time_values")
    plt.title("Prediction time for one verbatim (in s)")
    plt.show()
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

from typing import Dict 

def visualize(model_perf:Dict[str, Dict[str, int]]) -> None:
    perf_dict = {"metrics": [], "scores": [], "model_names": []}
    for model_name, performances in model_perf.items():
        for metric, score in performances.items():
            perf_dict["model_names"].append(model_name)
            perf_dict["metrics"].append(metric)
            perf_dict["scores"].append(score)
    perf_df = pd.DataFrame(perf_dict)
    sns.barplot(data=perf_df, x="metrics", y="scores", hue="model_names")
    plt.show() 
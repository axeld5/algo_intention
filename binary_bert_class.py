import numpy as np
import pandas as pd 
import evaluate 

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

if __name__ == "__main__":
    data = pd.read_csv("data/intent-detection-train.csv")
    train_texts, test_texts, train_labels, test_labels = train_test_split(data['text'], data['label'], test_size=0.33)
    # TO DO : OH encode labels, Make dataset usable by bert 
    train_dataset = None 
    test_dataset = None
    tokenizer = AutoTokenizer.from_pretrained("camembert-base")
    model = AutoModelForSequenceClassification.from_pretrained("camembert-base", num_labels=2)
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
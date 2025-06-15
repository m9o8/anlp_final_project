import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict, Dataset, load_from_disk
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TrainingArguments, Trainer,
                          EarlyStoppingCallback)
from transformers_interpret import SequenceClassificationExplainer
import evaluate, numpy as np, matplotlib.pyplot as plt
import torch, random
import pandas as pd
from collections import defaultdict



seed = 42

# Generalized class names - update this for your dataset
CLASS_NAMES = {
    0: "Human Needs", 1: "Operations & Transport", 2: "Chemistry & Metallurgy",
    3: "Textiles & Paper", 4: "Construction", 5: "Mechanical & Thermal Engineering",
    6: "Physics", 7: "Electricity", 8: "Cross-Tech Tagging"
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(seed)


class Metrics:
    def __init__(self):
        self.results = {}

    def run(self, y_true, y_pred, method_name, average='macro'):
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average)
        recall = recall_score(y_true, y_pred, average=average)
        f1 = f1_score(y_true, y_pred, average=average)

        # Store results
        self.results[method_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    def plot(self, custom_palette=None, save_plot_path=None):
        # Default color palette
        methods = list(self.results.keys())
        if custom_palette is None:
            custom_palette = sns.color_palette("Spectral", n_colors=len(methods))

        # Create subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']

        for i, metric in enumerate(metric_names):
            ax = axs[i // 2, i % 2]
            values = [self.results[m][metric.lower()] * 100 for m in methods]
            
            # Plot bars
            ax.bar(methods, values, color=custom_palette)
            ax.set_title(metric)
            ax.set_ylim(0, 100)

            # Annotate bars
            for j, v in enumerate(values):
                ax.text(j, v + 1, f"{v:.2f}", ha='center', va='bottom')

            # Rotate x-axis labels
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=45)

            # Style spines
            ax.spines['bottom'].set_color('black')
            ax.spines['left'].set_color('black')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Style ticks
            ax.tick_params(axis='y', colors='black', direction='in', length=5, width=1)
            ax.tick_params(axis='x', colors='black', direction='in', length=5, width=1)

        plt.tight_layout()
        if save_plot_path:
            plt.savefig(save_plot_path, bbox_inches='tight')
        plt.show()


### ======================================================================= ###
### ======================== MODEL TRAINING UTILS ========================= ###
### ======================================================================= ###

def keep_balanced_proportional(dataset, proportion=0.1, seed=42):
    df = dataset.to_pandas()
    rng = np.random.default_rng(seed)

    total_samples = round(len(df) * proportion)
    class_counts = df['labels'].value_counts()
    num_classes = len(class_counts)

    # Compute max balanced per class
    max_per_class = total_samples // num_classes

    # If the smallest class can't support that, balance isn't feasible
    if max_per_class <= class_counts.min():
        # Balanced sampling
        balanced_df = (
            df.groupby('labels')
            .apply(lambda x: x.sample(n=max_per_class, random_state=seed))
            .reset_index(drop=True)
        )
        return DatasetDict({"train": Dataset.from_pandas(balanced_df)})
    else:
        # Imbalanced fallback: sample randomly while keeping total size
        fallback_df = df.sample(n=total_samples, random_state=seed).reset_index(drop=True)
        return DatasetDict({"train": Dataset.from_pandas(fallback_df)})
    

metric_accuracy  = evaluate.load("accuracy")
metric_precision = evaluate.load("precision")
metric_recall    = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc  = metric_accuracy.compute(predictions=preds, references=labels)
    prec = metric_precision.compute(predictions=preds, references=labels, average="weighted", zero_division=0)
    rec  = metric_recall.compute(predictions=preds, references=labels, average="weighted", zero_division=0)

    # merge into a single dict that Trainer can log
    return {
        "accuracy":  acc["accuracy"],
        "precision": prec["precision"],
        "recall":    rec["recall"],
    }

def create_args_training(n_epochs, output_dir, batch_size=64):
  args = TrainingArguments(
      output_dir          = output_dir,
      eval_strategy       = "epoch",
      save_strategy       = "epoch",
      logging_strategy    = "epoch",
      logging_steps       = 50,
      learning_rate       = 2e-5,
      per_device_train_batch_size = batch_size,
      per_device_eval_batch_size  = batch_size,
      num_train_epochs    = n_epochs,
      weight_decay        = 0.01,
      load_best_model_at_end = True,
      metric_for_best_model = "eval_loss",
      save_total_limit    = 2,
      seed                = seed,
      report_to           = "none",  # Disable wandb
      fp16                = torch.cuda.is_available(), # Enable mixed precision training
      gradient_accumulation_steps = 2,
      dataloader_pin_memory=torch.cuda.is_available(),
  )
  return args
    

def train_and_predict(dataset, model, args, results_object, prop, model_ckpt, num_labels):
    sample_training = dataset["train"].shuffle(seed=42)
    sample_training = keep_balanced_proportional(sample_training, proportion=prop, seed=42)
    print(sample_training)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt,
        num_labels = num_labels,
        problem_type = "single_label_classification"
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = sample_training['train'],
        eval_dataset    = dataset["validation"].shuffle(seed=seed),
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)]
    )

    train_output = trainer.train()
    predictions = trainer.predict(dataset["test"])

    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    results_object.run(y_true, y_pred, f'D-BERT ({prop})', average='weighted')
    return results_object, y_pred.tolist(), y_true.tolist()
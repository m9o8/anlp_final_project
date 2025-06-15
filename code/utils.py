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

def create_args_training(n_epochs, output_dir):
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
    

def train_and_predict(dataset, model, args, results_object, prop):
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



### ======================================================================= ###
### ======================== MODEL UNDERSTANDING  ========================= ###
### ======================================================================= ###

def bert_interpretability_analysis(model, tokenizer, confused_examples, target_labels=None):
    """
    Generalized BERT interpretation for any label confusion

    Args:
        model: Trained model
        tokenizer: Model tokenizer
        confused_examples: List of (text, true_label, pred_label) tuples
        target_labels: Optional tuple of (true_label, pred_label) to filter for
    """

    # Initialize the explainer
    explainer = SequenceClassificationExplainer(model=model, tokenizer=tokenizer)
    analysis_results = []

    # Filter examples if target_labels specified
    if target_labels:
        true_target, pred_target = target_labels
        confused_examples = [(text, true, pred) for text, true, pred in confused_examples
                           if true == true_target and pred == pred_target]
        print(f"🎯 Focusing on: {CLASS_NAMES[true_target]} → {CLASS_NAMES[pred_target]}")

    for i, (text, true_label, pred_label) in enumerate(confused_examples[:5]):
        print(f"\n{'='*50}")
        print(f"CONFUSED EXAMPLE {i+1}")
        print(f"True: {CLASS_NAMES[true_label]} | Predicted: {CLASS_NAMES[pred_label]}")

        # Truncate text to fit model's max length
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > 500:
            tokens = tokens[:500]
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            print(f"⚠️  Text truncated to {len(tokens)} tokens")

        print(f"Text snippet: {text[:150]}...")

        try:
            # Get word attributions
            word_attributions = explainer(text)

            # Create visualization
            try:
                explainer.visualize(f"confusion_{CLASS_NAMES[true_label].replace(' ', '_')}_to_{CLASS_NAMES[pred_label].replace(' ', '_')}_{i+1}")
                print(f"📊 Visualization saved as HTML file")
            except Exception as viz_error:
                print(f"⚠️  HTML visualization failed: {viz_error}")

                # Fallback matplotlib plot
                words = [word for word, score in word_attributions[:15]]
                scores = [score for word, score in word_attributions[:15]]

                plt.figure(figsize=(12, 6))
                colors = ['red' if s < 0 else 'green' for s in scores]
                plt.barh(range(len(words)), scores, color=colors, alpha=0.7)
                plt.yticks(range(len(words)), words)
                plt.xlabel('Attribution Score')
                plt.title(f'Example {i+1}: {CLASS_NAMES[true_label]} → {CLASS_NAMES[pred_label]}')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()

            # Store attribution data
            attribution_data = []
            for word, score in word_attributions:
                attribution_data.append({
                    'word': word,
                    'attribution': score,
                    'abs_attribution': abs(score)
                })

            attribution_df = pd.DataFrame(attribution_data)
            attribution_df = attribution_df.sort_values('abs_attribution', ascending=False)

            print(f"\n🔍 Top 10 most influential words:")
            for _, row in attribution_df.head(10).iterrows():
                direction = "→" if row['attribution'] > 0 else "←"
                print(f"  {direction} '{row['word']}': {row['attribution']:.4f}")

            analysis_results.append({
                'example_id': i,
                'true_label': true_label,
                'pred_label': pred_label,
                'text': text,
                'attributions': attribution_df
            })

        except Exception as e:
            print(f"❌ Error processing example {i}: {e}")
            continue

    return analysis_results

def analyze_confusion_patterns_general(analysis_results, label_pair=None):
    """
    Generalized confusion pattern analysis for any label pair

    Args:
        analysis_results: Results from interpretability analysis
        label_pair: Optional (true_label, pred_label) tuple to focus on
    """

    print(f"\n{'='*60}")
    print("CONFUSION PATTERN ANALYSIS")
    print(f"{'='*60}")

    if label_pair:
        true_label, pred_label = label_pair
        # Filter for specific confusion direction
        filtered_results = [r for r in analysis_results
                          if r['true_label'] == true_label and r['pred_label'] == pred_label]

        if filtered_results:
            print(f"\n🎯 {CLASS_NAMES[true_label]} → {CLASS_NAMES[pred_label]} ({len(filtered_results)} examples):")

            # Find common influential words
            common_words = defaultdict(list)
            for result in filtered_results:
                top_words = result['attributions'].head(15)
                for _, row in top_words.iterrows():
                    if row['attribution'] > 0:  # Words pushing toward predicted class
                        common_words[row['word']].append(row['attribution'])

            # Show most misleading words
            avg_attribution = {word: np.mean(scores) for word, scores in common_words.items()
                              if len(scores) >= 1}

            if avg_attribution:
                sorted_words = sorted(avg_attribution.items(), key=lambda x: x[1], reverse=True)
                print(f"Words misleading model toward '{CLASS_NAMES[pred_label]}' prediction:")
                for word, avg_score in sorted_words[:10]:
                    freq = len(common_words[word])
                    print(f"  '{word}': {avg_score:.4f} (appears {freq}x)")
    else:
        # Analyze all confusion patterns
        confusion_groups = defaultdict(list)
        for result in analysis_results:
            key = (result['true_label'], result['pred_label'])
            confusion_groups[key].append(result)

        for (true_label, pred_label), results in confusion_groups.items():
            print(f"\n🎯 {CLASS_NAMES[true_label]} → {CLASS_NAMES[pred_label]} ({len(results)} examples):")

            common_words = defaultdict(list)
            for result in results:
                top_words = result['attributions'].head(10)
                for _, row in top_words.iterrows():
                    if row['attribution'] > 0:
                        common_words[row['word']].append(row['attribution'])

            avg_attribution = {word: np.mean(scores) for word, scores in common_words.items()
                              if len(scores) >= 1}

            if avg_attribution:
                sorted_words = sorted(avg_attribution.items(), key=lambda x: x[1], reverse=True)
                print(f"  Top misleading words:")
                for word, avg_score in sorted_words[:5]:
                    freq = len(common_words[word])
                    print(f"    • '{word}': {avg_score:.4f} ({freq}x)")

def find_most_confused_pairs(y_true, y_pred, top_n=5):
    """
    Automatically find the most confused label pairs

    Args:
        y_true: True labels
        y_pred: Predicted labels
        top_n: Number of top confused pairs to return

    Returns:
        List of (true_label, pred_label, count) tuples
    """

    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Find off-diagonal elements (misclassifications)
    confused_pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > 0:  # Misclassification
                confused_pairs.append((i, j, cm[i][j]))

    # Sort by confusion count
    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"🔍 Top {top_n} most confused label pairs:")
    for i, (true_label, pred_label, count) in enumerate(confused_pairs[:top_n]):
        print(f"  {i+1}. {CLASS_NAMES[true_label]} → {CLASS_NAMES[pred_label]}: {count} examples")

    return confused_pairs[:top_n]

def investigate_label_confusion_comprehensive(trainer, original_dataset, tokenizer,
                                            target_labels=None, auto_find_top=True):
    """
    Comprehensive label confusion analysis

    Args:
        trainer: Trained model trainer
        original_dataset: Dataset with text column
        tokenizer: Model tokenizer
        target_labels: Optional (true_label, pred_label) tuple to focus on
        auto_find_top: Whether to automatically find most confused pairs first
    """

    print("🔍 Starting Comprehensive Label Confusion Analysis...")
    print("=" * 60)

    # Get predictions
    predictions = trainer.predict(original_dataset["test"])
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    test_texts = original_dataset["test"]["text"]

    # Auto-find most confused pairs if requested
    if auto_find_top and target_labels is None:
        most_confused = find_most_confused_pairs(y_true, y_pred, top_n=5)

        # Ask user which pair to analyze (or automatically pick the top one)
        if most_confused:
            target_labels = (most_confused[0][0], most_confused[0][1])
            print(f"\n🎯 Auto-selected: {CLASS_NAMES[target_labels[0]]} → {CLASS_NAMES[target_labels[1]]}")

    # Find confused examples
    if target_labels:
        true_target, pred_target = target_labels
        confused_examples = [(text, true_label, pred_label)
                           for text, true_label, pred_label in zip(test_texts, y_true, y_pred)
                           if true_label == true_target and pred_label == pred_target]
        print(f"\nFound {len(confused_examples)} examples of {CLASS_NAMES[true_target]} → {CLASS_NAMES[pred_target]} confusion")
    else:
        # Analyze all misclassifications
        confused_examples = [(text, true_label, pred_label)
                           for text, true_label, pred_label in zip(test_texts, y_true, y_pred)
                           if true_label != pred_label]
        print(f"\nFound {len(confused_examples)} total misclassified examples")

    if len(confused_examples) == 0:
        print("✅ No confusion found!")
        return None, None

    # Run interpretability analysis
    print(f"\n🔬 Running attribution analysis on sample examples...")
    analysis_results = bert_interpretability_analysis(trainer.model, tokenizer,
                                                    confused_examples, target_labels)

    if analysis_results:
        # Analyze patterns
        print(f"\n📊 Analyzing confusion patterns...")
        analyze_confusion_patterns_general(analysis_results, target_labels)

        return analysis_results, confused_examples
    else:
        print("❌ No successful analyses completed")
        return None, None

# Potential usages

# 1. Analyze specific label pair
def analyze_specific_confusion(trainer, dataset, tokenizer, true_label, pred_label):
    """Analyze confusion between two specific labels"""
    return investigate_label_confusion_comprehensive(
        trainer, dataset, tokenizer,
        target_labels=(true_label, pred_label),
        auto_find_top=False
    )

# 2. Auto-find and analyze top confusions
def analyze_top_confusions(trainer, dataset, tokenizer):
    """Automatically find and analyze most confused pairs"""
    return investigate_label_confusion_comprehensive(
        trainer, dataset, tokenizer,
        target_labels=None,
        auto_find_top=True
    )

# 3. Analyze all confusions
def analyze_all_confusions(trainer, dataset, tokenizer):
    """Analyze patterns across all label confusions"""
    return investigate_label_confusion_comprehensive(
        trainer, dataset, tokenizer,
        target_labels=None,
        auto_find_top=False
    )
"""GLUE dataset loading, preprocessing, and metric helpers.

Each GLUE task is trained independently because tasks use different label spaces
and classification heads.  By default the complete original train split is used
for training, while the labeled validation split is used both for model
selection and final evaluation because the official GLUE test labels are hidden.
"""

import json
import math
import os

import numpy as np
from datasets import load_dataset, load_from_disk


GLUE_TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def normalize_glue_task(task_name):
    task_name = str(task_name).lower().strip()
    if task_name not in GLUE_TASK_TO_KEYS:
        supported = ", ".join(GLUE_TASK_TO_KEYS)
        raise ValueError(
            f"Unknown GLUE task {task_name!r}. Supported tasks: {supported}."
        )
    return task_name


def get_glue_task_info(task_name):
    """Return ``(task_name, num_labels, is_regression)``."""
    task_name = normalize_glue_task(task_name)
    is_regression = task_name == "stsb"
    num_labels = 1 if is_regression else (3 if task_name == "mnli" else 2)
    return task_name, num_labels, is_regression


def load_glue_data(
    task_name,
    tokenizer,
    max_length=128,
    dataset_name="nyu-mll/glue",
    cache_dir=None,
    max_train_samples=None,
    max_eval_samples=None,
    pad_to_max_length=False,
    validation_ratio=0.0,
    seed=42,
):
    """Load/tokenize GLUE and create train/validation/test datasets.

    With ``validation_ratio=0``, the complete original train split is used for
    training and the original labeled validation split is used both for model
    selection and final evaluation.  A positive ratio optionally restores an
    internal holdout from train.
    """
    task_name, num_labels, is_regression = get_glue_task_info(task_name)
    validation_ratio = float(validation_ratio)
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError(
            f"validation_ratio must be in [0, 1), got {validation_ratio}."
        )
    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[task_name]
    if os.path.isdir(dataset_name):
        raw_datasets = load_from_disk(dataset_name)
    else:
        raw_datasets = load_dataset(
            dataset_name,
            task_name,
            cache_dir=cache_dir,
        )

    tokenizer_limit = getattr(tokenizer, "model_max_length", max_length)
    if tokenizer_limit is None or tokenizer_limit > 1_000_000:
        tokenizer_limit = max_length
    max_length = min(int(max_length), int(tokenizer_limit))
    padding = "max_length" if pad_to_max_length else False

    def preprocess_function(examples):
        texts = (examples[sentence1_key],)
        if sentence2_key is not None:
            texts = (examples[sentence1_key], examples[sentence2_key])
        return tokenizer(
            *texts,
            padding=padding,
            max_length=max_length,
            truncation=True,
        )

    tokenized = raw_datasets.map(
        preprocess_function,
        batched=True,
        desc=f"Tokenizing GLUE/{task_name}",
    )

    train_pool = tokenized["train"]
    if max_train_samples is not None:
        count = min(int(max_train_samples), len(train_pool))
        if count <= 1:
            raise ValueError(
                "max_train_samples must leave at least two examples before "
                "the train/validation split."
            )
        train_pool = train_pool.select(range(count))

    validation_dataset = None
    if validation_ratio > 0.0:
        split_kwargs = {
            "test_size": validation_ratio,
            "seed": int(seed),
        }
        if not is_regression:
            split_kwargs["stratify_by_column"] = "label"
        train_validation = train_pool.train_test_split(**split_kwargs)
        train_dataset = train_validation["train"]
        validation_dataset = train_validation["test"]
        if max_eval_samples is not None:
            count = min(int(max_eval_samples), len(validation_dataset))
            validation_dataset = validation_dataset.select(range(count))
    else:
        train_dataset = train_pool

    # The original labeled validation split is held out as the final test set.
    if task_name == "mnli":
        split_names = {
            "mnli": "validation_matched",
            "mnli-mm": "validation_mismatched",
        }
    else:
        split_names = {task_name: "validation"}

    eval_datasets = {}
    for eval_name, split_name in split_names.items():
        dataset = tokenized[split_name]
        if max_eval_samples is not None:
            count = min(int(max_eval_samples), len(dataset))
            dataset = dataset.select(range(count))
        eval_datasets[eval_name] = dataset

    if validation_dataset is None:
        validation_dataset = eval_datasets[task_name]

    return {
        "task_name": task_name,
        "num_labels": num_labels,
        "is_regression": is_regression,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "eval_datasets": eval_datasets,
        "validation_dataset": validation_dataset,
        "test_datasets": eval_datasets,
    }


def _accuracy(predictions, references):
    return float(np.mean(predictions == references))


def _binary_f1(predictions, references):
    predictions = np.asarray(predictions) == 1
    references = np.asarray(references) == 1
    true_positive = int(np.sum(predictions & references))
    false_positive = int(np.sum(predictions & ~references))
    false_negative = int(np.sum(~predictions & references))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2.0 * true_positive / denominator


def _matthews_correlation(predictions, references):
    predictions = np.asarray(predictions) == 1
    references = np.asarray(references) == 1
    tp = int(np.sum(predictions & references))
    tn = int(np.sum(~predictions & ~references))
    fp = int(np.sum(predictions & ~references))
    fn = int(np.sum(~predictions & references))
    denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    return 0.0 if denominator == 0 else (tp * tn - fp * fn) / denominator


def _pearson_correlation(predictions, references):
    predictions = np.asarray(predictions, dtype=np.float64)
    references = np.asarray(references, dtype=np.float64)
    predictions = predictions - predictions.mean()
    references = references - references.mean()
    denominator = np.linalg.norm(predictions) * np.linalg.norm(references)
    return 0.0 if denominator == 0 else float(
        np.dot(predictions, references) / denominator
    )


def _rankdata(values):
    """Return average ranks for ties without requiring SciPy."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def compute_glue_metrics(task_name, predictions, references):
    """Compute the official-style metrics for one GLUE validation task."""
    task_name = normalize_glue_task(task_name.replace("-mm", ""))
    predictions = np.asarray(predictions)
    references = np.asarray(references)

    if task_name == "stsb":
        pearson = _pearson_correlation(predictions, references)
        spearman = _pearson_correlation(
            _rankdata(predictions), _rankdata(references)
        )
        return {
            "pearson": pearson,
            "spearmanr": spearman,
            "combined_score": (pearson + spearman) / 2.0,
        }
    if task_name == "cola":
        return {
            "matthews_correlation": _matthews_correlation(
                predictions, references
            )
        }

    accuracy = _accuracy(predictions, references)
    if task_name in {"mrpc", "qqp"}:
        f1 = _binary_f1(predictions, references)
        return {
            "accuracy": accuracy,
            "f1": f1,
            "combined_score": (accuracy + f1) / 2.0,
        }
    return {"accuracy": accuracy}


def make_glue_compute_metrics(task_name, is_regression=False):
    task_name = normalize_glue_task(task_name)

    def compute_metrics(eval_prediction):
        predictions = eval_prediction.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = (
            np.squeeze(predictions)
            if is_regression
            else np.argmax(predictions, axis=-1)
        )
        return compute_glue_metrics(
            task_name, predictions, eval_prediction.label_ids
        )

    return compute_metrics


def evaluate_glue_trainer(trainer, glue_data, output_dir, logger=None):
    """Evaluate the held-out original GLUE validation splits as final tests."""
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = {}
    for eval_name, eval_dataset in glue_data["eval_datasets"].items():
        trainer.compute_metrics = make_glue_compute_metrics(
            eval_name.replace("-mm", ""), glue_data["is_regression"]
        )
        metrics = trainer.evaluate(
            eval_dataset=eval_dataset,
            metric_key_prefix=f"eval_{eval_name}",
        )
        serializable = {key: float(value) for key, value in metrics.items()}
        all_metrics[eval_name] = serializable
        if logger is not None:
            logger.info(f"[GLUE/{eval_name}] {serializable}")

    output_path = os.path.join(output_dir, "glue_results.json")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2, ensure_ascii=False)
    return all_metrics

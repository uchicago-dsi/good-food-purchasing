"""Defines MultiTaskTrainer and accessory functions for training multi-task text classifier for Center for Good Food Purchasing"""

import json
import logging
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn.functional import sigmoid
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import EvalPrediction

from cgfp.inference.inference import test_inference


def compute_metrics(
    pred: Any,
    model: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Extracts the predictions and labels for each task and computes metrics.

    Note:
        For distilbert:
        - pred.predictions: (num_heads, batch_size, num_classes) (note: this is a list)
        - pred.label_ids: (batch_size, num_columns, 1) (note: this is a tensor)

        For roberta:
        - pred.predictions: tuple of shape (num_heads, batch_size)
          - First element is a list of shape: (num_heads, batch_size, num_classes)
          - Second element is a numpy array of shape: (batch_size, hidden_dim) (output logits)

    Args:
        pred: The predictions and labels from the model, structure depends on the model type.
        model: The name of the model being used ("distilbert" or "roberta").
        threshold: The threshold to apply for binary classification tasks
        basic_type_idx: The index of the basic type to use for evaluation

    Returns:
        A dictionary containing the computed metrics for each task.
    """
    batch_size = pred.label_ids.shape[0]
    num_tasks = len(model.classification_heads)

    accuracies = {}
    f1_scores = {}
    precisions = {}
    recalls = {}

    # Note: Distilbert returns just preds, Roberta returns preds and logits
    if model.config.base_model_type == "distilbert":
        predictions = pred.predictions
    elif model.config.base_model_type == "roberta":
        predictions = pred.predictions[0]

    for i, task in enumerate(model.classification_heads.keys()):
        lbl = pred.label_ids[:, i, 0].tolist()
        if i != model.subtypes_head_idx:
            # Handle non-subtype tasks
            pred_lbl = predictions[i].argmax(-1)
            accuracies[task] = accuracy_score(lbl, pred_lbl)
            f1_scores[task] = f1_score(
                lbl, pred_lbl, average="weighted", zero_division=np.nan
            )  #  Use weighted for multi-class classification
            precisions[task] = precision_score(lbl, pred_lbl, average="weighted", zero_division=np.nan)
            recalls[task] = recall_score(lbl, pred_lbl, average="weighted", zero_division=np.nan)

    # Note: Handle subtype predictions - this is a multilabel task so kind of messy

    # Get all indices with probs above a threshold
    # Note: preds_subtype is a list with dimensions (batch_size, num_subtype_classes)
    preds_subtype = (
        (sigmoid(torch.tensor(predictions[int(model.subtypes_head_idx)])) > threshold).int().tolist()
    )  # TODO: threshold should come from the model config maybe?

    # Create a zeros matrix with dimensions (batch_size, num_subtype_classes)
    num_subtype_classes = len(model.decoders["Sub-Types"])
    lbls_subtype = torch.zeros((batch_size, num_subtype_classes), dtype=int)
    # Change each index for each example for each subtype col to 1
    for idx in model.subtype_data_indices:
        lbls = pred.label_ids[:, idx, 0]
        lbls_subtype[torch.arange(batch_size), lbls] = 1
    # Don't compute accuracy for "None" for multilabel task - set to 0
    lbls_subtype[torch.arange(batch_size), int(model.none_subtype_idx)] = 0

    accuracies["Sub-Types"] = accuracy_score(lbls_subtype, preds_subtype)
    f1_scores["Sub-Types"] = f1_score(lbls_subtype, preds_subtype, average="weighted", zero_division=np.nan)
    precisions["Sub-Types"] = precision_score(lbls_subtype, preds_subtype, average="weighted", zero_division=np.nan)
    recalls["Sub-Types"] = recall_score(lbls_subtype, preds_subtype, average="weighted", zero_division=np.nan)

    basic_type_accuracy = accuracies["Basic Type"]
    mean_accuracy = sum(accuracies.values()) / num_tasks
    mean_f1_score = sum(f1_scores.values()) / num_tasks

    return {
        "mean_accuracy": mean_accuracy,
        "accuracies": accuracies,
        "mean_f1_score": mean_f1_score,
        "f1_scores": f1_scores,
        "precisions": precisions,
        "recalls": recalls,
        "basic_type_accuracy": basic_type_accuracy,
    }


class SaveBestModelCallback(TrainerCallback):
    """A custom callback for saving the best model during training based on a specified evaluation metric.

    Args:
        model: The model being trained, which will be saved when the specified metric improves.
        tokenizer: The tokenizer associated with the model, to be saved alongside the model.
        device: The device (e.g., 'cpu' or 'cuda') on which the model is being trained.
        best_model_metric: The metric used to determine the best model. The model is saved if this metric improves.
        eval_prompt: The evaluation prompt or function used during the evaluation phase.
    """

    def __init__(self, model: Any, tokenizer: Any, device: str, best_model_metric: str, eval_prompt: Any):
        """Initializes the SaveBestModelCallback."""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.best_metric = -float("inf")
        self.best_model_metric = best_model_metric
        self.best_epoch = None
        self.eval_prompt = eval_prompt

    def on_evaluate(self, args: Any, state: Any, control: Any, metrics: dict[str, Any] = None, **kwargs: Any) -> None:
        """Callback function executed during the evaluation phase.

        Args:
            self: The instance of the class.
            args: Arguments related to the evaluation process.
            state: The current state of the training process.
            control: Control flow for the training loop.
            metrics: A dictionary of evaluation metrics. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        if state.epoch is not None:
            logging.info(f"### EPOCH: {int(state.epoch)} ###")
        # Note: "eval_" is prepended to the keys in metrics
        current_metric = metrics["eval_" + self.best_model_metric]
        pretty_metrics = json.dumps(metrics, indent=4)
        logging.info(pretty_metrics)
        if state.epoch is not None:
            if state.epoch % 5 == 0 or current_metric > self.best_metric:
                prompt = "frozen peas and carrots"
                test_inference(self.model, self.tokenizer, prompt, self.device)
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.best_epoch = state.epoch
            logging.info(f"Best model updated at epoch: {state.epoch} with metric ({current_metric})")

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Callback function executed at the end of training.

        Args:
            self: The instance of the class.
            args: Arguments related to the training process.
            state: The current state of the training process.
            control: Control flow for the training loop.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        if self.best_epoch is not None:
            logging.info(f"The best model was saved from epoch: {self.best_epoch}")
            logging.info(f"The best result was {self.best_model_metric}: {self.best_metric}")


class MultiTaskTrainer(Trainer):
    """A custom trainer class for handling multi-task training with the Hugging Face Trainer API.

    Args:
        *args: Positional arguments passed to the base Trainer class.
        **kwargs: Keyword arguments passed to the base Trainer class.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initializes the MultiTaskTrainer with any arguments required by the base Trainer class."""
        super().__init__(*args, **kwargs)
        # Note: Model dicts are different — kind of ugly way to get the weights we want
        if self.model.config.model_type == "distilbert":
            transformer = "transformer"
            query = "q_lin"
        elif self.model.config.model_type == "roberta":
            transformer = "encoder"
            query = "self.query"

        def get_query(layer):
            if self.model.config.model_type == "roberta":
                return layer.attention.self.query
            return getattr(layer.attention, query)

        self.logging_params = {
            "First Attention Layer Q": get_query(getattr(self.model.llm, transformer).layer[0]).weight,
            "Last Attention Layer Q": get_query(getattr(self.model.llm, transformer).layer[-1]).weight,
            "Basic Type Classification Head": self.model.classification_heads["Basic Type"][0].weight,
            "Food Product Group Classification Head": self.model.classification_heads["Food Product Group"][0].weight,
            "Sub-Types Classification Head": self.model.classification_heads["Sub-Types"][0].weight,
        }

    def compute_metrics(self, p: EvalPrediction) -> dict[str, Any]:
        """Computes and returns evaluation metrics based on model predictions.

        Args:
            self: The instance of the class.
            p: An EvalPrediction object containing predictions and label_ids.

        Returns:
            A dictionary containing the computed metrics.
        """
        return compute_metrics(p, self.model)

    def log_gradients(self, name: str, param: Any) -> None:
        """Computes and logs the sum of gradients for a given parameter.

        Args:
            self: The instance of the class.
            name: The name of the parameter whose gradients are being logged.
            param: The parameter object, which may or may not have gradients.

        Returns:
            None
        """
        value = param.grad.sum() if param.grad is not None else 0
        logging.info(f"Gradient sum for {name}: {value}")

    def training_step(self, model: Any, inputs: Any, interval: int = 5) -> torch.Tensor:
        """Performs a single training step and logs gradients every 5 epochs.

        Args:
            self: The instance of the class.
            model: The model being trained.
            inputs: The input data for the training step.
            interval: The epoch interval on which to log gradients

        Returns:
            A tensor representing the loss for the current training step.
        """
        loss = super().training_step(model, inputs)

        if (self.state.epoch) % interval == 0 and self.state.epoch != 0:
            logging.info(f"## Gradients at Epoch {int(self.state.epoch)} ##")
            for name, param in self.logging_params.items():
                self.log_gradients(name, param)
        return loss

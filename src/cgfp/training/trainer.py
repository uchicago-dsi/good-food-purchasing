import json
import logging

import torch
from cgfp.constants.training_constants import BASIC_TYPE_IDX
from cgfp.inference.inference import test_inference
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.functional import sigmoid
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import EvalPrediction


def compute_metrics(pred, model, threshold=0.5, basic_type_idx=BASIC_TYPE_IDX):
    """Extract the predictions and labels for each task

    pred.predictions: (num_heads, batch_size, num_classes) » note: this is a list
    pred.label_ids: (batch_size, num_columns, 1) » note: this is a tensor
    """
    batch_size = pred.label_ids.shape[0]
    num_tasks = len(model.classification_heads)

    accuracies = {}
    f1_scores = {}

    # Handle non-subtype predictions
    for i, task in enumerate(model.classification_heads.keys()):
        lbl = pred.label_ids[:, i, 0].tolist()
        if i != model.subtypes_head_idx:
            # Handle non-subtype tasks
            pred_lbl = pred.predictions[i].argmax(-1)
            accuracies[task] = accuracy_score(lbl, pred_lbl)
            f1_scores[task] = f1_score(
                lbl, pred_lbl, average="weighted"
            )  #  Use weighted for multi-class classification

    # Handle subtype predictions - this is a multilabel task so kind of messy
    num_subtype_classes = len(model.decoders["Sub-Types"])

    # Get all indices with probs above a threshold
    # Note: preds_subtyps is a list with dimensions (batch_size, num_subtype_classes)
    preds_subtype = (
        (sigmoid(torch.tensor(pred.predictions[int(model.subtypes_head_idx)])) > threshold).int().tolist()
    )  # TODO: threshold should come from the model config maybe?

    # Create a zeros matrix with dimensions (batch_size, num_subtype_classes)
    lbls_subtype = torch.zeros((batch_size, num_subtype_classes), dtype=int)
    # Change each index for each example for each subtype col to 1
    for idx in model.subtype_data_indices:
        lbls = pred.label_ids[:, idx, 0]
        lbls_subtype[torch.arange(batch_size), lbls] = 1
    # Don't compute accuracy for "None" for multilabel task - set to 0
    lbls_subtype[torch.arange(batch_size), int(model.none_subtype_idx)] = 0

    accuracies["Sub-Types"] = accuracy_score(lbls_subtype, preds_subtype)
    f1_scores["Sub-Types"] = f1_score(lbls_subtype, preds_subtype, average="weighted")

    # # TODO: this is the old way...
    # num_tasks = len(pred.predictions)
    # preds = [
    #     pred.predictions[i].argmax(-1) for i in range(len(model.classification_heads))
    # ]  #  (num_heads, batch_size)
    # labels = [pred.label_ids[:, i, 0].tolist() for i in range(len(model.config.columns))]  #  (num_cols, batch_size)
    # accuracies = {}
    # f1_scores = {}
    # for i, task in enumerate(zip(preds, labels)):
    #     pred, lbl = task
    #     accuracies[i] = accuracy_score(lbl, pred)
    #     f1_scores[i] = f1_score(lbl, pred, average="weighted")  # Use weighted for multi-class classification

    basic_type_accuracy = accuracies["Basic Type"]
    mean_accuracy = sum(accuracies.values()) / num_tasks
    mean_f1_score = sum(f1_scores.values()) / num_tasks

    return {
        "mean_accuracy": mean_accuracy,
        "accuracies": accuracies,
        "mean_f1_score": mean_f1_score,
        "f1_scores": f1_scores,
        "basic_type_accuracy": basic_type_accuracy,
    }


class SaveBestModelCallback(TrainerCallback):
    def __init__(self, model, tokenizer, device, best_model_metric, eval_prompt):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.best_metric = -float("inf")
        self.best_model_metric = best_model_metric
        self.best_epoch = None
        self.eval_prompt = eval_prompt

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        logging.info(f"### EPOCH: {int(state.epoch)} ###")
        # Note: "eval_" is prepended to the keys in metrics
        current_metric = metrics["eval_" + self.best_model_metric]
        pretty_metrics = json.dumps(metrics, indent=4)
        logging.info(pretty_metrics)
        if state.epoch % 5 == 0 or current_metric > self.best_metric:
            prompt = "frozen peas and carrots"
            test_inference(self.model, self.tokenizer, prompt, self.device)
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.best_epoch = state.epoch
            logging.info(f"Best model updated at epoch: {state.epoch} with metric ({current_metric})")

    def on_train_end(self, args, state, control, **kwargs):
        if self.best_epoch is not None:
            logging.info(f"The best model was saved from epoch: {self.best_epoch}")
            logging.info(f"The best result was {self.best_model_metric}: {self.best_metric}")


# TODO: How does logging work here?
class MultiTaskTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: This won't work if the model isn't distilbert...
        self.logging_params = {
            "First Attention Layer Q": self.model.distilbert.transformer.layer[0].attention.q_lin.weight,
            "Last Attention Layer Q": self.model.distilbert.transformer.layer[-1].attention.q_lin.weight,
            "Basic Type Classification Head": self.model.classification_heads["Basic Type"][0].weight,
            "Food Product Group Classification Head": self.model.classification_heads["Food Product Group"][0].weight,
            "Sub-Types Classification Head": self.model.classification_heads["Sub-Types"][0].weight,
        }

    def compute_metrics(self, p: EvalPrediction):
        return compute_metrics(p, self.model)

    def log_gradients(self, name, param):
        """Safely compute the sum of gradients for a given parameter."""
        value = param.grad.sum() if param.grad is not None else 0
        logging.info(f"Gradient sum for {name}: {value}")

    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)

        if (self.state.epoch) % 5 == 0 and self.state.epoch != 0:
            logging.info(f"## Gradients at Epoch {int(self.state.epoch)} ##")
            for name, param in self.logging_params.items():
                self.log_gradients(name, param)
        return loss

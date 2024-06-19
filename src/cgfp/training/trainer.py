import logging
import json

from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainerCallback

from cgfp.inference.inference import test_inference
from cgfp.constants.training_constants import BASIC_TYPE_IDX

def compute_metrics(pred, basic_type_idx=BASIC_TYPE_IDX):
    """
    Extract the predictions and labels for each task

    TODO: organize this info in a docstring
    len(pred.predictions) » 2
    len(pred.predictions[0]) » 20
    len(pred.predictions[0][0]) » 6 (number of classes)
    len(pred.predictions[1][0]) » 29 (number of classes)
    Also...why is this 20 and not the batch size?

    len(pred.label_ids) » 2
    This comes in a list of length 20 with a 2D label for each example?
    array([[ 5],
       [26]])
    """
    num_tasks = len(pred.predictions)
    preds = [pred.predictions[i].argmax(-1) for i in range(num_tasks)]
    labels = [pred.label_ids[:, i, 0].tolist() for i in range(num_tasks)]

    accuracies = {}
    f1_scores = {}
    for i, task in enumerate(zip(preds, labels)):
        pred, lbl = task
        accuracies[i] = accuracy_score(lbl, pred)
        f1_scores[i] = f1_score(lbl, pred, average='weighted')  # Use weighted for multi-class classification

    basic_type_accuracy = accuracies[basic_type_idx]
    mean_accuracy = sum(accuracies.values()) / num_tasks
    mean_f1_score = sum(f1_scores.values()) / num_tasks

    return {"mean_accuracy": mean_accuracy, "accuracies": accuracies, "mean_f1_score": mean_f1_score, "f1_scores": f1_scores, "basic_type_accuracy": basic_type_accuracy}


class SaveBestModelCallback(TrainerCallback):
    def __init__(self, model, tokenizer, device, best_model_metric, eval_prompt):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.best_metric = -float('inf')
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
            "Basic Type Classification Head": self.model.classification_heads['Basic Type'][0].weight,
            "Food Product Group Classification Head": self.model.classification_heads['Food Product Group'][0].weight,
            "Sub-Type 1 Classification Head": self.model.classification_heads['Sub-Type 1'][0].weight,
        }

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

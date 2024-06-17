import json

import torch
import torch.nn as nn
import torch.nn.functional as F

# from collections import OrderedDict

from transformers import (
    # DistilBertForSequenceClassification,
    # DistilBertTokenizerFast,
    # TrainingArguments,
    # Trainer,
    PreTrainedModel,
    DistilBertConfig,
    DistilBertModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from cgfp.constants.training_constants import BASIC_TYPE_IDX, FPG_IDX

# TODO: Clean this up
class FocalLoss(nn.Module):
    # TODO: add documentation for the alpha and gamma parameters
    def __init__(self, alpha=None, gamma=2.0, num_classes=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # TODO: set alpha based on class frequency » we should maybe calculate this during data processing?
        if alpha is None:
            self.alpha = torch.ones(num_classes) / num_classes
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float)
        if self.alpha is not None:
            if self.alpha.size(0) != num_classes:
                raise ValueError("Alpha vector size must match number of classes.")
        self.alpha = self.alpha.cuda() if self.alpha is not None else None

    def forward(self, inputs, targets):
        # Compute the cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get the probabilities for the classes that are actually true
        pt = torch.exp(-ce_loss)
        
        # Calculate alpha for each example
        if self.alpha is not None:
            alpha = self.alpha.gather(0, targets.data)
        else:
            alpha = 1

        # Calculate the Focal Loss
        focal_loss = alpha * ((1 - pt) ** self.gamma) * ce_loss

        return focal_loss.mean()


class MultiTaskConfig(DistilBertConfig):
    def __init__(
        self,
        classification="linear",
        decoders=None,
        columns=None,
        counts=None,
        fpg_idx=FPG_IDX,
        basic_type_idx=BASIC_TYPE_IDX,
        inference_masks=None,
        loss="cross_entropy",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.decoders = decoders
        self.columns = columns
        self.classification = classification  # choices are "linear" or "mlp"
        self.loss = loss
        self.counts = counts
        # self.num_categories_per_task = [len(v) for v in json.loads(self.counts).values()]

        # TODO: can maybe get these indexes from the columns
        self.fpg_idx = fpg_idx
        self.basic_type_idx = basic_type_idx
        self.inference_masks = inference_masks


class MultiTaskModel(PreTrainedModel):
    config_class = MultiTaskConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.config = config
        self.distilbert = DistilBertModel(config)

        # Note: Need to store inference masks and counts as JSON in config, so need to initialize them
        self.initialize_inference_masks()
        self.initialize_counts()

        self.initialize_classification_heads()
        self.initialize_losses()
        self.set_attached_heads(self.config.columns)

    def initialize_classification_heads(self):
        if self.config.classification == "mlp":
            self.classification_heads = nn.ModuleDict({
                task_name: nn.Sequential(
                    nn.Linear(self.config.dim, self.config.dim // 2),
                    nn.ReLU(),
                    nn.Dropout(self.config.seq_classif_dropout),
                    nn.Linear(self.config.dim // 2, num_categories),
                ) for task_name, num_categories in zip(self.config.columns, self.num_categories_per_task)
            })
        elif self.config.classification == "linear":
            self.classification_heads = nn.ModuleDict({
                task_name: nn.Sequential(
                    nn.Linear(self.config.dim, num_categories),
                    nn.Dropout(self.config.seq_classif_dropout),
                ) for task_name, num_categories in zip(self.config.columns, self.num_categories_per_task)
            })

    def initialize_losses(self):
        self.losses = []
        if self.config.loss == "focal":
            # TODO: Results here are unstable/bad — probably not actually correct
            for task, counts in self.counts.items():
                counts = torch.tensor(counts, dtype=torch.float)
                total = counts.sum()
                alpha = (1 / counts) * (total / len(counts)) # Use the inverse frequency
                # alpha /= alpha.sum()  # Normalize to sum to 1
                self.losses.append(FocalLoss(num_classes=len(counts), alpha=alpha))
        else:
            for task, counts in self.counts.items():
                self.losses.append(nn.CrossEntropyLoss())

    def initialize_inference_masks(self):
        self.inference_masks = {key: torch.tensor(value) for key, value in json.loads(self.config.inference_masks).items()}

    def initialize_counts(self):
        self.counts = json.loads(self.config.counts)
        self.num_categories_per_task = [len(v) for k, v in self.counts.items()]

    def set_attached_heads(self, heads_to_attach):
        """Set which heads should have their inputs attached to the computation graph. Allows for controlling the finetuning of the model."""
        if not all(head in self.classification_heads for head in heads_to_attach):
            raise ValueError("One or more specified heads do not exist in the model.")
        self.attached_heads = heads_to_attach

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=False,
    ):
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]

        # TODO: write actual documentation of what's happening here if this works
        logits = []
        # TODO: uh....how should I actually do this?
        # unfreeze_heads = ["Food Product Group", "Food Product Category", "Basic Type"]
        for i, item in enumerate(self.classification_heads.items()):
            head, classifier = item
            if head not in self.attached_heads:
                logits.append(classifier(pooled_output.detach()))
            else:
                logits.append(classifier(pooled_output))

        loss = None
        losses = []
        # TODO: Fragile...fix later
        if labels is not None:
            for i, output in enumerate(zip(
                logits, labels.squeeze().transpose(0, 1)
            )):  # trust me
                logit, label = output
                losses.append(self.losses[i](logit, label.view(-1)))
            loss = sum(losses)

        output = (logits,) + distilbert_output[
            1:
        ]  # TODO why activations? see https://github.com/huggingface/transformers/blob/6f316016877197014193b9463b2fd39fa8f0c8e4/src/transformers/models/distilbert/modeling_distilbert.py#L824

        if not return_dict:
            return (loss,) + output if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )
    
if __name__ == "__main__":
    pass
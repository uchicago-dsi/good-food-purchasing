import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    DistilBertConfig,
    DistilBertModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

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
        num_categories_per_task=None,
        decoders=None,
        columns=None,
        counts=None,
        fpg_idx=0,
        basic_type_idx=2,
        inference_masks=None,
        loss="focal",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_categories_per_task = num_categories_per_task
        self.decoders = decoders
        self.columns = columns
        self.classification = classification  # choices are "linear" or "mlp"
        self.fpg_idx = fpg_idx
        self.basic_type_idx=basic_type_idx
        self.inference_masks=inference_masks
        self.loss=loss
        self.counts=counts


class MultiTaskModel(PreTrainedModel):
    config_class = MultiTaskConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.num_categories_per_task = config.num_categories_per_task
        self.decoders = config.decoders
        self.columns = config.columns
        self.classification = config.classification
        self.fpg_idx = config.fpg_idx  # index for the food product group task
        self.basic_type_idx = config.basic_type_idx
        self.inference_masks = {key: torch.tensor(value) for key, value in json.loads(config.inference_masks).items()}
        self.loss = config.loss
        self.counts = json.loads(config.counts)
        self.losses = []

        if self.loss == "focal":
            for task, counts in self.counts.items():
                counts = torch.tensor(counts, dtype=torch.float)
                total = counts.sum()
                alpha = (1 / counts) * (total / len(counts)) # Use the inverse frequency
                alpha /= alpha.sum()  # Normalize to sum to 1
                self.losses.append(FocalLoss(num_classes=len(counts), alpha=alpha))
        else:
            for task, counts in self.counts.items():
                self.losses.append(nn.CrossEntropy())

        if self.classification == "mlp":
            # TODO: wait...the config.dim should be downsampled here probably...
            self.classification_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(config.dim, config.dim // 2),
                        nn.ReLU(),
                        nn.Dropout(config.seq_classif_dropout),
                        nn.Linear(config.dim // 2, num_categories),
                    )
                    for num_categories in self.num_categories_per_task
                ]
            )
        elif self.classification == "linear":
            self.classification_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(config.dim, num_categories),
                        nn.Dropout(config.seq_classif_dropout),
                    )
                    for num_categories in self.num_categories_per_task
                ]
            )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
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

        logits = [classifier(pooled_output) for classifier in self.classification_heads]

        loss = None
        if labels is not None:
            losses = []
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


# TODO: Maybe need to subclass the Trainer to modify the loss and backprop process?# Only backpropagate the loss from the first head to the shared layers
# loss1.backward(retain_graph=True)  # retain_graph is needed if multiple losses affect the same parameters

# # Detach the shared layer outputs to prevent gradients from loss2 affecting shared layers
# shared_output = model.shared(input).detach().requires_grad_()
# output2 = model.head2(shared_output)

# # Recalculate loss2 with detached shared output
# loss2 = criterion(output2, target2)
# loss2.backward()  # This will only update head2
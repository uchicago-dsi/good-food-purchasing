import json

import torch
import torch.nn as nn

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

# TODO: Actually implement FocalLoss
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, num_classes=None):
        super(FocalLoss, self).__init__()
        assert (alpha is None and num_classes is not None) or (alpha is not None and num_classes is None), "Specify alpha or num_classes"
        if alpha is None:
            # Automatic calculation based on class frequency
            self.alpha = torch.ones(num_classes) / num_classes  # This could be a more sophisticated function of class frequency
        else:
            # Predefined alpha values
            self.alpha = torch.tensor(alpha)
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.long()
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


class MultiTaskConfig(DistilBertConfig):
    def __init__(
        self,
        classification="linear",
        num_categories_per_task=None,
        decoders=None,
        columns=None,
        fpg_idx=0,
        basic_type_idx=2,
        inference_masks=None,
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


        # TODO:
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
            loss_fct = nn.CrossEntropyLoss()
            losses = []
            for logit, label in zip(
                logits, labels.squeeze().transpose(0, 1)
            ):  # trust me
                losses.append(loss_fct(logit, label.view(-1)))
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
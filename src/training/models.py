import json

import torch
import torch.nn as nn

from collections import OrderedDict

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, TrainingArguments, Trainer, PreTrainedModel, DistilBertConfig, DistilBertModel
from transformers.modeling_outputs import SequenceClassifierOutput

class MultiTaskConfig(DistilBertConfig):
    def __init__(self, classification="linear", num_categories_per_task=None, decoders=None, columns=None, **kwargs):
        super().__init__(**kwargs)
        self.num_categories_per_task = num_categories_per_task
        self.decoders = decoders
        self.columns = columns
        self.classification = classification # choices are "linear" or "mlp"

class MultiTaskModel(PreTrainedModel):
    config_class = MultiTaskConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.num_categories_per_task = config.num_categories_per_task
        self.decoders = config.decoders
        self.columns = config.columns
        self.classification = config.classification

        # TODO: 
        if self.classification == "mlp":
            # TODO: wait...the config.dim should be downsampled here probably...
            self.classification_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.dim, config.dim // 2),
                    nn.ReLU(),
                    nn.Dropout(config.seq_classif_dropout),
                    nn.Linear(config.dim // 2, num_categories)
                ) for num_categories in self.num_categories_per_task
            ])
        elif self.classification == "linear":
            self.classification_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.dim, num_categories),
                    nn.Dropout(config.seq_classif_dropout)
                ) for num_categories in self.num_categories_per_task
            ])

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
            for logit, label in zip(logits, labels.squeeze().transpose(0, 1)): # trust me
                losses.append(
                    loss_fct(logit, label.view(-1))
                )
            loss = sum(losses)

        output = (logits,) + distilbert_output[1:] # TODO why activations? see https://github.com/huggingface/transformers/blob/6f316016877197014193b9463b2fd39fa8f0c8e4/src/transformers/models/distilbert/modeling_distilbert.py#L824
    
        if not return_dict:
            return (loss,) + output if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions
        )
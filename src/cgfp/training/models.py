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
    RobertaModel,
    RobertaTokenizerFast,
    RobertaConfig,
)
from transformers.modeling_outputs import SequenceClassifierOutput

# TODO: probably need a factory function to switch the class dynamically
# class MultiTaskConfig(DistilBertConfig):
class MultiTaskConfig(RobertaConfig):
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
        self.basic_type_idx = basic_type_idx
        self.inference_masks = inference_masks
        # TODO: this is for RoBERTa
        self.dim = self.hidden_size


class MultiTaskModel(PreTrainedModel):
    config_class = MultiTaskConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        # self.distilbert = DistilBertModel(config)
        self.llm = RobertaModel(config)
        self.num_categories_per_task = config.num_categories_per_task
        self.decoders = config.decoders
        self.columns = config.columns
        self.classification = config.classification
        self.fpg_idx = config.fpg_idx  # index for the food product group task
        self.basic_type_idx = config.basic_type_idx
        self.inference_masks = {key: torch.tensor(value) for key, value in json.loads(config.inference_masks).items()}

        # Attributes that differ between DistilBERT and RoBERTa
        self.dropout_rate = getattr(config, 'seq_classif_dropout', getattr(config, 'hidden_dropout_prob', 0.2))
        # TODO: move config.dim to here

        if self.classification == "mlp":
            # TODO: wait...the config.dim should be downsampled here probably...
            self.classification_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(config.dim, config.dim // 2),
                        nn.ReLU(),
                        nn.Dropout(self.dropout_rate),
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
        llm_output = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_state = llm_output[0]
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

        output = (logits,) + llm_output[
            1:
        ]  # TODO why activations? see https://github.com/huggingface/transformers/blob/6f316016877197014193b9463b2fd39fa8f0c8e4/src/transformers/models/distilbert/modeling_distilbert.py#L824

        if not return_dict:
            return (loss,) + output if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=llm_output.hidden_states,
            attentions=llm_output.attentions,
        )

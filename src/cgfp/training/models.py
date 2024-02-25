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

from openai import OpenAI

from cgfp.config_tags import GROUP_TAGS, CATEGORY_TAGS, GROUP_CATEGORY_VALIDATION
from cgfp.config_prompt import HINTS


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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_categories_per_task = num_categories_per_task
        self.decoders = decoders
        self.columns = columns
        self.classification = classification  # choices are "linear" or "mlp"
        self.fpg_idx = fpg_idx
        self.basic_type_idx = basic_type_idx
        self.inference_masks = inference_masks


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
        self.inference_masks = {
            key: torch.tensor(value)
            for key, value in json.loads(config.inference_masks).items()
        }

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


class GPTAPI:
    def __init__(self):
        self.client = OpenAI()

    def forward(self, example, model="gpt-3.5-turbo-0125"):
        content = f"""
                    We are trying to help a small non-profit label food items that come from invoices and purchase orders using a hierarchical tagging system. 

                    The tags are organized as follows and should be output for each item in csv format:
                    Food Product Group, Food Product Category, Product Name, Basic Type, Sub-Type 1, Sub-Type 2, Flavor/Cut, Shape, Skin, Seed/Bone	Processing, Cooked/Cleaned, WG/WGR, Dietary Concern, Additives, Dietary Accommodation, Frozen	Packaging, Commodity

                    Here are the allowed tags for each food product group in dictionary format:\n
                    {str(GROUP_TAGS)}
                    """
        content += f"""\n
                    The food product category is a level 'below' the food product group. Here are the allowed food product categories for each food product group:\n
                    {str(GROUP_CATEGORY_VALIDATION)}
                    """
        content += f"""\n
                    Here are some additional allowed tags for each food product category:\n
                    {str(CATEGORY_TAGS)}
                    """
        content += """\n
                Each column can have at most one tag and columns (other than food product group, food product category, and basic type) can all be blank.

                Basic type is the main food type present in the food. Sub type 1 and sub type 2 are used to provide additional detail about the food item if necessary. Something like the kind of cheese, the kind of cereal (oat, corn, or rice), etc.

                Please include an empty dictionary entry if the output is none for a specific category. Remember that Food Product Group, Food Product Category, and Basic Type must always have an entry. None and empty strings are not valid. 

                Below you will see few examples of the task. 

                ONLY OUTPUT THE DICTIONARY ADD A CONFIDENCE SCORE! Do not explain yourself or add context. This will be used in an API so it's very important that you only return a dictionary.

                Also, please be extra careful to make sure that the : is in the right spot on every line. If there's no :, the response will generate an error and won't be able to be parsed correctly.

                Prompt: beef patty 2 oz	
                Output: {
                "Food Product Group": "Meat", 
                "Food Product Category": "Beef", 
                "Basic Type": "beef", 
                "Sub-Type 1": None, 
                "Sub-Type 2": None, 
                "Flavor/Cut": None, 
                "Shape": "patty", 
                "Skin": None, 
                "Seed/Bone": None,
                "Processing": None, 
                "Cooked/Cleaned": None, 
                "WG/WGR": None, 
                "Dietary Concern": None, 
                "Additives": None, 
                "Dietary Accommodation": None, 
                "Frozen": None,
                "Packaging": None, 
                "Commodity": None
                }

                Prompt: CHEESE CUP ULTIMATE CHEDDAR
                Output: {
                "Food Product Group": "Milk & Dairy", 
                "Food Product Category": "Cheese", 
                "Basic Type": "sauce", 
                "Sub-Type 1": "cheese", 
                "Sub-Type 2": "cheddar", 
                "Flavor/Cut": None, 
                "Shape": "patty", 
                "Skin": None, 
                "Seed/Bone": None,
                "Processing": None, 
                "Cooked/Cleaned": None, 
                "WG/WGR": None, 
                "Dietary Concern": None, 
                "Additives": None, 
                "Dietary Accommodation": None, 
                "Frozen": None,
                "Packaging": "ss", 
                "Commodity": None
                }"""
        content += f"""Here's some hints on how to do this task better: \n
                    {HINTS}"""
        content += f"""Prompt: {example}\n
        Output:
        """

        stream = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            stream=True,
        )

        response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content

        try:
            # replace None with null since JSON expects null
            parsed_response = json.loads(response.replace("None", "null"))
            return parsed_response
        except Exception as e:
            print(f"Response was not valid JSON. {e} \n Response: {response}")
            return {}  # Can be converted into a

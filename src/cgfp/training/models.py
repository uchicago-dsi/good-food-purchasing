"""Contains custom multitask models and configurations for the CGFP project"""

import json
import logging
from typing import Any, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    DistilBertModel,
    PretrainedConfig,
    PreTrainedModel,
    RobertaModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput


# TODO: This doesn't actually work very well...
class FocalLoss(nn.Module):
    """Focal Loss function, commonly used for addressing class imbalance in classification tasks.

    Args:
        alpha: A balancing factor for the class weights. Can be a list of weights for each class or None.
        gamma: The focusing parameter that adjusts the rate at which easy examples are down-weighted.
        num_classes: The number of classes in the classification task. Used to shape the loss function.
    """

    def __init__(self, alpha=None, gamma=2.0, num_classes=None):
        """Initializes the FocalLoss with alpha and gamma parameters."""
        super().__init__()
        self.gamma = gamma
        # TODO: set alpha based on class frequency
        if alpha is None:
            self.alpha = torch.ones(num_classes) / num_classes
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float)
        if self.alpha is not None:
            if self.alpha.size(0) != num_classes:
                raise ValueError("Alpha vector size must match number of classes.")
        self.alpha = self.alpha.cuda() if self.alpha is not None else None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass of the Focal Loss function.

        Args:
            self: The instance of the FocalLoss class.
            inputs: The predicted logits or probabilities from the model, typically of shape (batch_size, num_classes).
            targets: The ground truth labels, typically of shape (batch_size).

        Returns:
            A tensor representing the computed Focal Loss.
        """
        # Compute the cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

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


class MultiTaskConfig(PretrainedConfig):
    """Configuration class for multi-task models, extending from PretrainedConfig.

    Args:
        base_model_type: The type of model to use (e.g., "distilbert")
        classification: The type of classification head to use (e.g., "linear")
        decoders: A dictionary of decoders to use for each task.
        columns: A dictionary mapping task names to data columns.
        counts: A dictionary containing the counts of classes for each task.
        inference_masks: A dictionary of masks used for inference, indicating valid outputs for each task.
        loss: The loss function to use (e.g., "cross_entropy")
        combine_subtypes: A boolean indicating whether to combine subtypes during training
        subtype_orders: The order in which subtypes should be processed, if applicable.
        **kwargs: Additional keyword arguments passed to the PretrainedConfig.
    """

    def __init__(
        self,
        base_model_type="roberta",
        classification="linear",
        decoders=None,
        columns=None,
        counts=None,
        inference_masks=None,
        loss="cross_entropy",
        combine_subtypes=False,
        subtype_orders=None,
        **kwargs,
    ):
        """Initializes the PreTrainedConfig with the passed parameters"""
        super().__init__(**kwargs)
        self.base_model_type = base_model_type

        # Note: distilbert has some unconventionally named config options
        if base_model_type == "distilbert":
            self.attribute_map = {
                "hidden_size": "dim",
                "num_attention_heads": "n_heads",
                "num_hidden_layers": "n_layers",
                "classifier_dropout": "seq_classif_dropout",
            }
        else:
            self.attribute_map = {}

        self.decoders = decoders
        self.columns = columns
        self.classification = classification
        self.loss = loss
        self.counts = counts
        self.inference_masks = inference_masks
        self.combine_subtypes = combine_subtypes

    def __getattr__(self, name: str) -> Any:
        """Handles attribute access for non-standard attribute names by looking up mapped names.

        Args:
            self: The instance of the class.
            name: The name of the attribute being accessed.

        Returns:
            The value of the requested attribute, either from the mapped name or by calling the superclass's __getattr__.

        Raises:
            AttributeError: If the attribute is not found in either the mapped names or the superclass.
        """
        if name in self.attribute_map:
            mapped_name = self.attribute_map[name]
            return getattr(self.base_config, mapped_name)
        else:
            try:
                return super().__getattr__(name)
            except AttributeError as e:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'") from e

    def __setattr__(self, name: str, value: Any) -> None:
        """Handles setting attributes, mapping non-standard attribute names to their corresponding base configuration attributes.

        Args:
            self: The instance of the class.
            name: The name of the attribute being set.
            value: The value to assign to the attribute.

        Returns:
            None
        """
        if name in self.attribute_map:
            mapped_name = self.attribute_map[name]
            setattr(self.base_config, mapped_name, value)
        else:
            super().__setattr__(name, value)


class MultiTaskModel(PreTrainedModel):
    """A multi-task learning model based on the DistilBert or RoBERTa architecture for handling multiple classification tasks.

    This model allows for training and inference on multiple tasks simultaneously by attaching multiple
    classification heads to a shared DistilBert backbone. It supports different types of classification heads
    (MLP or linear) and loss functions (CrossEntropy or Focal Loss). The model is highly configurable via the
    MultiTaskConfig class and allows for dynamic attachment and detachment of classification heads.

    Attributes:
        config: The configuration object containing model parameters.
        llm: The DistilBert or RoBERTa model serving as the backbone.
        classification_heads: The classification tasks.
        loss_fns: The loss functions for each task.
        inference_masks: The inference masks for each task. These are the allowed categories based on tag structure.
        counts: The counts of each class for each task.
        num_categories_per_task: The number of categories per task.
        attached_heads: The classification heads that should be attached to the computation graph.
    """

    config_class = MultiTaskConfig

    def __init__(self, config: MultiTaskConfig, *args, **kwargs):
        """Initialize the multi-task model with the given configuration."""
        super().__init__(config)
        self.config = config

        if self.config.model_type == "distilbert":
            self.llm = DistilBertModel(config)
        elif self.config.model_type == "roberta":
            self.llm = RobertaModel(config)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")

        # Note: Need to store some config objects as JSON, so need to initialize them
        self.initialize_inference_masks()
        self.initialize_tasks()

        self.initialize_losses()

        # Note: When loading from checkpoint, the model runs __init__ then loads weights after model
        # is initialized — so this is safe for loading weights
        self.initialize_classification_heads()

        # Note: Initialize with all heads attached. Can change this directly by invoking the method.
        self.set_attached_heads(self.decoders.keys())

    def initialize_classification_heads(self) -> None:
        # TODO: Do we want to specify the head here?
        """Initialize the classification heads based on the configuration."""
        if self.config.classification == "mlp":
            self.classification_heads = nn.ModuleDict(
                {
                    task_name: nn.Sequential(
                        nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
                        nn.ReLU(),
                        nn.Dropout(self.config.classifier_dropout),
                        nn.Linear(self.config.hidden_size // 2, len(decoder)),
                    )
                    for task_name, decoder in self.decoders.items()
                }
            )
        elif self.config.classification == "linear":
            self.classification_heads = nn.ModuleDict(
                {
                    task_name: nn.Sequential(
                        nn.Linear(self.config.hidden_size, len(decoder)),
                        nn.Dropout(self.config.classifier_dropout),
                    )
                    for task_name, decoder in self.decoders.items()
                }
            )

    def initialize_losses(self) -> None:
        """Initialize the loss functions for each task based on the configuration."""
        self.loss_fns = {}
        for task in self.decoders.keys():
            if task == "Sub-Types":
                logging.info(f"Using BCEWithLogitsLoss for {task}")
                self.loss_fns[task] = nn.BCEWithLogitsLoss()
            else:
                logging.info(f"Using CrossEntropyLoss for {task}")
                self.loss_fns[task] = nn.CrossEntropyLoss()

    def initialize_inference_masks(self) -> None:
        """Initialize the inference masks from the configuration."""
        self.inference_masks = {
            key: torch.tensor(value) for key, value in json.loads(self.config.inference_masks).items()
        }

    def initialize_tasks(self) -> None:
        """Initialize the counts and number of categories per task from the configuration."""
        self.decoders = dict(self.config.decoders)

        self.subtype_data_indices = [idx for task, idx in self.config.columns.items() if "Sub-Type" in task]
        self.none_subtype_idx = next(idx for idx, subtype in self.decoders["Sub-Types"].items() if subtype == "None")
        self.subtypes_head_idx = list(self.decoders.keys()).index("Sub-Types")

        self.counts = json.loads(self.config.counts)

    def set_attached_heads(self, heads_to_attach: Union[str, List[str]]) -> None:
        """Set which heads should have their inputs attached to the computation graph. Allows for controlling the finetuning of the model."""
        logging.info(f"Running model with {heads_to_attach} heads attached to the computation graph...")
        if isinstance(heads_to_attach, str):
            heads_to_attach = {heads_to_attach}
        else:
            heads_to_attach = set(heads_to_attach)
        if not all(head in self.classification_heads for head in heads_to_attach):
            raise ValueError("One or more specified heads do not exist in the model.")
        self.attached_heads = heads_to_attach

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        labels: torch.Tensor = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = False,
    ) -> SequenceClassifierOutput:
        """Perform a forward pass through the model."""
        base_model_output = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_state = base_model_output[0]
        pooled_output = hidden_state[:, 0]

        # Note: For each forward pass, we detach specified classification heads
        # in order to prevent gradients flowing to the base model
        logits = []
        for task_name, classifier in self.classification_heads.items():
            if task_name in self.attached_heads:
                logits.append(classifier(pooled_output))
            else:
                logits.append(classifier(pooled_output.detach()))

        loss = None
        losses = []
        if labels is not None:
            for i, (task, logit) in enumerate(zip(self.classification_heads.keys(), logits)):
                if task != "Sub-Types":
                    formatted_labels = labels.squeeze().transpose(0, 1)[i].view(-1)
                    losses.append(self.loss_fns[task](logit, formatted_labels))
                else:
                    # Handle sub-types separately for multi-label classification
                    batch_size = labels.shape[0]
                    subtype_labels = []
                    # TODO: Need to fix this
                    for idx in self.subtype_data_indices:
                        subtype_labels.append(labels.squeeze().transpose(0, 1)[idx].view(-1))
                    # TODO:
                    all_labels = torch.stack(subtype_labels)  # Shape: (# of subtype columns, batch_size)
                    target = torch.zeros(
                        (batch_size, len(self.decoders["Sub-Types"])), device=self.device
                    )  # Shape: (batch size, num classes)

                    # Create multi-label target tensor
                    # TODO: I think this can be done better...
                    for labels in all_labels:
                        # Note: labels are sub-type labels for one sub-type column for a whole batch
                        # Shape: (batch_size,)
                        for batch_idx, lbl in enumerate(labels):
                            # We don't want the multilabel head predicting "None" so exclude that idx
                            if lbl != int(self.none_subtype_idx):
                                # print(f"Adding {self.decoders['Sub-Types'][str(lbl.item())]} as subtype")
                                target[batch_idx, lbl] = 1
                    multilabel_loss = self.loss_fns["Sub-Types"](logit, target)
                    # print("multilabel loss", multilabel_loss)
                    losses.append(multilabel_loss)

            loss = sum(losses)

        output = (
            (logits,) + base_model_output[1:]
        )  # Note: why activations? see https://github.com/huggingface/transformers/blob/6f316016877197014193b9463b2fd39fa8f0c8e4/src/transformers/models/distilbert/modeling_distilbert.py#L824

        if not return_dict:
            return (loss,) + output if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=base_model_output.hidden_states,
            attentions=base_model_output.attentions,
        )


if __name__ == "__main__":
    pass

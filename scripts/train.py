import sys
import logging
import json
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import polars as pl
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import datasets
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
from datasets import Dataset

from cgfp.inference.inference import inference
from cgfp.training.models import MultiTaskConfig, MultiTaskModel

logging.basicConfig(level=logging.INFO)

### Setup

MODEL_NAME = "distilbert-base-uncased"
TEXT_FIELD = "Product Type"
# TODO: This is kind of fragile to changes in capitalization, etc.
# Fix this so it doesn't matter if the columns are the same case or not
LABELS = [
    "Food Product Group",
    "Food Product Category",
    "Primary Food Product Category",
    "Basic Type",
    "Sub-Type 1",
    "Sub-Type 2",
    "Flavor/Cut",
    "Shape",
    "Skin",
    "Seed/Bone",
    "Processing",
    "Cooked/Cleaned",
    "WG/WGR",
    "Dietary Concern",
    "Additives",
    "Dietary Accommodation",
    "Frozen",
    "Packaging",
    "Commodity",
]
# These indeces are used to set up inference filtering
FPG_IDX = LABELS.index("Food Product Group")
BASIC_TYPE_IDX = LABELS.index("Basic Type")

# TODO: add args to MODEL_PATH and logging path
# TODO: make model path name more descriptive
MODEL_PATH = (
    f"/net/projects/cgfp/model-files/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
)

SMOKE_TEST = True
SAVE_BEST = True

FREEZE_LAYERS = False
FREEZE_MLPS = False

if SMOKE_TEST:
    MODEL_PATH += "-smoke-test"


def read_data(data_path):
    df = pl.read_csv(data_path, infer_schema_length=1, null_values=["NULL"]).lazy()
    df_cleaned = df.select(TEXT_FIELD, *LABELS)
    # TODO: We should probably drop rows that contain integers
    # For now, just convert to string so we can run this
    # Reminder: polars syntax is different than pandas syntax
    df_cleaned = df_cleaned.with_columns(pl.col(TEXT_FIELD).cast(pl.Utf8))
    df_cleaned = df_cleaned.filter(pl.col(TEXT_FIELD).is_not_null())
    df_cleaned = df_cleaned.fill_null("None")
    return df_cleaned


### Data prep

# Training

def compute_metrics(pred):
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


    mean_accuracy = sum(accuracies.values()) / num_tasks
    mean_f1_score = sum(f1_scores.values()) / num_tasks

    return {"mean_accuracy": mean_accuracy, "accuracies": accuracies, "mean_f1_score": mean_f1_score, "f1_scores": f1_scores}


if __name__ == "__main__":

    logging.info(f"MODEL_PATH : {MODEL_PATH}")

    # Setup

    logging.info("Starting")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info(f"Training with device : {device}")

    logging.info(f"Using base model : {MODEL_NAME}")
    logging.info(f"Predicting based on input field : {TEXT_FIELD}")
    logging.info(f"Predicting categorical fields : {LABELS}")

    # Data preparation

    # TODO: Set this up so data file comes out of data pipeline
    # TODO: Fix this also so that the arguments make more sense...
    data_path = sys.argv[1] if len(sys.argv) > 1 else "/net/projects/cgfp/data/clean/clean_CONFIDENTIAL_CGFP_bulk_data_073123.csv"
    logging.info(f"Reading data from path : {data_path}")
    df_train = read_data(data_path)
    df_eval = read_data("/net/projects/cgfp/data/clean/clean_New_Raw_Data_030724.csv")
    df_combined = pl.concat([df_train, df_eval]) # combine training and eval so we have all valid outputs for evaluation

    encoders = {}
    for column in LABELS:
        encoder = LabelEncoder()
        encoder.fit_transform(df_combined.select(column).collect().to_numpy().ravel())
        encoders[column] = encoder

    # Create decoders to save to model config
    # Huggingface is picky...so a list of tuples seems like the best bet
    # for saving to the config.json in a way that doesn't break when loaded
    decoders = []
    for col, encoder in encoders.items():
        decoding_dict = {
            f"{index}": label for index, label in enumerate(encoder.classes_)
        }
        decoders.append((col, decoding_dict))
        logging.info(f"{col}: {len(encoder.classes_)} classes")

    # Save valid basic types for each food product group
    # Reminder: polars syntax is different than pandas
    inference_masks = {}
    basic_types = df_combined.select("Basic Type").unique().collect()['Basic Type'].to_list()
    for fpg in df_combined.select('Food Product Group').unique().collect()['Food Product Group']:
        valid_basic_types = df_combined.filter(pl.col("Food Product Group") == fpg).select("Basic Type").unique().collect()['Basic Type'].to_list()

        # logging to inspect basic types
        # logging.info(f"{fpg} basic types")
        # logging.info(valid_basic_types)

        basic_type_indeces = encoders['Basic Type'].transform(valid_basic_types)
        mask = np.zeros(len(basic_types))
        mask[basic_type_indeces] = 1
        inference_masks[fpg] = mask.tolist()

    logging.info("Preparing dataset")
    train_dataset = Dataset.from_pandas(df_train.collect().to_pandas())
    eval_dataset = Dataset.from_pandas(df_eval.collect().to_pandas())

    if SMOKE_TEST:
        train_dataset = train_dataset.select(range(1000))

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        tokenized_inputs = tokenizer(
            batch[TEXT_FIELD], padding="max_length", truncation=True, max_length=100
        )
        tokenized_inputs["labels"] = [
            encoders[label].transform([batch[label]]) for label in LABELS
        ]
        return tokenized_inputs

    # TODO: there's a better way to do this...
    # Should probably put most (or all) of this in a function and handle the config stuff after the dataset is setup
    train_dataset = train_dataset.map(tokenize)
    eval_dataset = eval_dataset.map(tokenize)
    for i in range(5):
        logging.info(train_dataset[i])

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    # train_dataset = train_dataset.train_test_split(test_size=0.2)
    logging.info("Dataset is prepared")

    logging.info(f"Structure of the dataset : {train_dataset}")
    logging.info(f"Sample record from the dataset : {train_dataset[0]}")

    # Training

    logging.info("Instantiating model")

    # TODO: set this up so that classification can be passed via args
    classification = "mlp"
    distilbert_model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased"
    )
    num_categories_per_task = [len(v.classes_) for k, v in encoders.items()]
    config = MultiTaskConfig(
        num_categories_per_task=num_categories_per_task,
        decoders=decoders,
        columns=LABELS,
        classification=classification,
        fpg_idx=FPG_IDX,
        basic_type_idx=BASIC_TYPE_IDX,
        inference_masks=json.dumps(inference_masks),
        **distilbert_model.config.to_dict(),
    )
    model = MultiTaskModel(config)
    logging.info("Model instantiated")

    epochs = 5 if SMOKE_TEST else 40

    # TODO: add an arg for freezing layers
    # Freeze all layers
    if FREEZE_LAYERS:
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze classification heads
        for param in model.classification_heads.parameters():
            param.requires_grad = True

    if FREEZE_MLPS:
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze attention heads and layernorm
        for name, param in model.named_parameters():
            if "attention" in name or "output" in name or "layer_norm" in name:
                param.requires_grad = True

        # Unfreeze classification heads
        for param in model.classification_heads.parameters():
            param.requires_grad = True

        for name, param in model.named_parameters():
            print(f"{name} is {'frozen' if not param.requires_grad else 'unfrozen'}")


    # TODO: set this up to come from args
    lr = 0.001

    # TODO: Training logs argument doesn't seem to work. Logs are in the normal logging folder?
    # TODO: Add info to logging file name
    training_args = TrainingArguments(
        # TODO: come up with may to manage storage space for checkpoints
        output_dir="/net/projects/cgfp/checkpoints",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=100,
        logging_dir="./training-logs",
        max_grad_norm=1.0,
    )

    if SAVE_BEST:
        training_args.load_best_model_at_end = True
        training_args.metric_for_best_model = "mean_accuracy"
        training_args.greater_is_better = True

    # TODO: set this up to come from args
    lr = 0.001
    adamW = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-5, weight_decay=0.1)
    scheduler = CosineAnnealingWarmRestarts(adamW, T_0=2000, T_mult=1, eta_min=lr*0.1)

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        # train_dataset=combined_dataset,
        train_dataset=train_dataset,
        # eval_dataset=dataset["test"],
        eval_dataset=eval_dataset,
        optimizers=(adamW, scheduler),  # Pass optimizers here (rather than training_args) for more fine-grained control
    )

    logging.info("Training...")
    trainer.train()

    logging.info("Training complete")
    logging.info(f"Validation Results: {trainer.evaluate()}")

    if not SMOKE_TEST:
        logging.info("Saving the model")
        model.save_pretrained(MODEL_PATH)
        tokenizer.save_pretrained(MODEL_PATH)

    logging.info("Complete!")

    prompt = "frozen peas and carrots"
    legible_preds = inference(model, tokenizer, prompt, device)
    logging.info(f"Example output for 'frozen peas and carrots': {legible_preds}")
import os
import logging
import json
from datetime import datetime
import argparse
import yaml
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import polars as pl
import numpy as np
import pandas as pd

from typing import Dict, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset

import wandb

from cgfp.inference.inference import inference, inference_handler
from cgfp.training.models import MultiTaskConfig, MultiTaskModel
from cgfp.config_training import LABELS

SCRIPT_DIR = Path(__file__).resolve().parent


def read_data(input_col, labels, data_path):
    # Note: polars syntax is different than pandas syntax
    df = pl.read_csv(data_path, infer_schema_length=1, null_values=["NULL"]).lazy()
    for col in [input_col, "Food Product Group", "Food Product Category", "Primary Food Product Category"]:
        prev_height = df.collect().shape[0]
        df = df.filter(pl.col(col).is_not_null())
        new_height = df.collect().shape[0]
        logging.info(f"Excluded {prev_height - new_height} rows due to null values in '{col}'.")
    df_cleaned = df.select(input_col, *labels)
    # TODO: This maybe needs to be done to each column? 
    # Make sure every row is correctly encoded as string
    df_cleaned = df_cleaned.with_columns(pl.col(input_col).cast(pl.Utf8))
    # TODO: make this come from function args
    # TODO: should we use FPC to fix nulls for PFPC? This is a pipeline question
    df_cleaned = df_cleaned.fill_null("None")
    return df_cleaned

def test_inference(model, tokenizer, prompt, device="cuda:0"):
    normalized_name = inference(model, tokenizer, prompt, device, combine_name=True)
    logging.info(f"Example output for 'frozen peas and carrots': {normalized_name}")
    preds_dict = inference(model, tokenizer, prompt, device)
    pretty_preds = json.dumps(preds_dict, indent=4)
    logging.info(pretty_preds)

# TODO: Move to trainer file
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

    basic_type_accuracy = accuracies[BASIC_TYPE_IDX]
    mean_accuracy = sum(accuracies.values()) / num_tasks
    mean_f1_score = sum(f1_scores.values()) / num_tasks

    return {"mean_accuracy": mean_accuracy, "accuracies": accuracies, "mean_f1_score": mean_f1_score, "f1_scores": f1_scores, "basic_type_accuracy": basic_type_accuracy}


if __name__ == "__main__":
    with open(SCRIPT_DIR / "config_train.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Setup
    TEXT_FIELD = config['data']['text_field']
    SMOKE_TEST = config['config']['smoke_test']
    SAVE_BEST = config['model']['save_best']
    MODEL_NAME = config['model']['model_name']
    RESET_CLASSIFICATION_HEADS = config['model']['reset_classification_heads']

    # Logging
    run_name = f"{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    if SMOKE_TEST:
        run_name += "-smoke-test"
        os.environ["WANDB_DISABLED"] = "true"
    MODEL_SAVE_PATH = Path(config['model']['model_dir']) / run_name

    LOGGING_DIR = SCRIPT_DIR.parent / "logs/"
    LOG_FILE = f"{LOGGING_DIR}{run_name}.log"
    logging.basicConfig(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

    transformers_logger = logging.getLogger('transformers')
    transformers_logger.setLevel(logging.INFO)
    transformers_logger.addHandler(file_handler)

    if not SMOKE_TEST:
        wandb.init(project='cgfp', name=run_name)

    # Config
    classification = config['model']['classification']
    loss = config['model']['loss']

    # Hyperparameters
    lr = config['training']['lr']
    epochs = config['training']['epochs'] if not SMOKE_TEST else 6
    train_batch_size = config['training']['train_batch_size']
    eval_batch_size = config['training']['eval_batch_size']

    ### SETUP ###
    logging.info(f"MODEL_PATH : {MODEL_SAVE_PATH}")
    logging.info("Starting")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info(f"Training with device : {device}")
    logging.info(f"Using base model : {MODEL_NAME}")
    logging.info(f"Predicting based on input field : {TEXT_FIELD}")
    logging.info(f"Predicting categorical fields : {LABELS}")

    ### DATA PREP ###
    # These indeces are used to set up inference filtering
    FPG_IDX = LABELS.index("Food Product Group")
    BASIC_TYPE_IDX = LABELS.index("Basic Type")
    DATA_DIR = Path(config['data']['data_dir'])
    TRAIN_DATA_PATH = DATA_DIR / config['data']['train_filename']
    EVAL_DATA_PATH = DATA_DIR / config['data']['eval_filename']

    logging.info(f"Reading training data from path : {TRAIN_DATA_PATH}")
    df_train = read_data(TEXT_FIELD, LABELS, TRAIN_DATA_PATH)
    logging.info(f"Reading eval data from path : {TRAIN_DATA_PATH}")
    df_eval = read_data(TEXT_FIELD, LABELS, EVAL_DATA_PATH)

    df_combined = pl.concat([df_train, df_eval]) # combine training and eval so we have all valid outputs for evaluation

    encoders = {}
    counts = {}
    counts_sheets = {}
    for column in LABELS:
        # Create encoders (including all labels from training and eval sets)
        # Note: sort this so that the order is consistent
        unique_categories = df_combined.select(column).unique().sort(column).collect().to_numpy().ravel()
        encoder = LabelEncoder()
        encoder.fit(unique_categories)
        encoders[column] = encoder

        # Get counts for each category in the training set for focal loss
        counts_df = df_train.group_by(column).agg(pl.len().alias('count')).sort(column)

        logging.info(f"Categories for {column}")
        with pl.Config(tbl_rows=-1):
            logging.info(counts_df.collect())

        # Fill 0s for categories that aren't in the training set
        full_counts_df = pl.DataFrame({column: unique_categories})
        full_counts_df = full_counts_df.join(counts_df.collect(), on=column, how='left').fill_null(0)
        counts_sheets[column] = full_counts_df
        counts[column] = full_counts_df['count'].to_list()

    # Create decoders to save to model config
    decoders = []
    for col, encoder in encoders.items():
        # Note: Huggingface is picky with config.json...a list of tuples works
        decoding_dict = {
            f"{index}": label for index, label in enumerate(encoder.classes_)
        }
        decoders.append((col, decoding_dict))

    # Save valid basic types for each food product group
    inference_masks = {}
    basic_types = df_combined.select("Basic Type").unique().collect()['Basic Type'].to_list()
    for fpg in df_combined.select('Food Product Group').unique().collect()['Food Product Group']:
        # Note: polars syntax is different than pandas
        valid_basic_types = df_combined.filter(pl.col("Food Product Group") == fpg).select("Basic Type").unique().collect()['Basic Type'].to_list()
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

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    logging.info("Datasets are prepared")
    logging.info(f"Structure of the dataset : {train_dataset}")
    logging.info(f"Sample record from the dataset : {train_dataset[0]}")

    ### TRAINING ###

    logging.info("Instantiating model")
    checkpoint = config['model']['checkpoint']
    FREEZE_BASE = config['model']['freeze_base']
    MODEL_SAVE_PATH = MODEL_SAVE_PATH / run_name


    # TODO: Stop using this now that we have the counts. Use counts object instead in models.py
    num_categories_per_task = [len(v.classes_) for k, v in encoders.items()]

    if checkpoint is None:
        distilbert_model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)

        multi_task_config = MultiTaskConfig(
            num_categories_per_task=num_categories_per_task,
            decoders=decoders,
            columns=LABELS,
            classification=classification,
            fpg_idx=FPG_IDX,
            basic_type_idx=BASIC_TYPE_IDX,
            inference_masks=json.dumps(inference_masks),
            counts=json.dumps(counts),
            loss=loss,
            **distilbert_model.config.to_dict(),
        )
        model = MultiTaskModel(multi_task_config)
    else:
        # Note: ignore_mismatched_sizes since we are often loading from checkpoints with different numbers of categories
        model = MultiTaskModel.from_pretrained(checkpoint, ignore_mismatched_sizes=True)

        # Note: If the data has changed, we need to update the model config
        model.config.decoders = decoders
        model.config.num_categories_per_task = num_categories_per_task

        # Note: inference masks and counts are finicky due to the way they are saved in the config
        # Need to save them as JSON and initialize them in the model
        model.config.inference_masks = json.dumps(inference_masks)
        model.config.counts = json.dumps(counts)
        model.initialize_inference_masks()
        model.initialize_counts()

    model.save_pretrained(MODEL_SAVE_PATH)

    if RESET_CLASSIFICATION_HEADS:
        model.initialize_classification_heads()

    if FREEZE_BASE:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the classification heads
        for name, head in model.classification_heads.items():
            for param in head.parameters():
                param.requires_grad = True

    # TODO: Add something to config to set this
    model.set_attached_heads(LABELS)

    logging.info("Model instantiated")

    # TODO: Move this to a separate file
    class SaveBestModelCallback(TrainerCallback):
        def __init__(self, best_model_metric):
            self.best_metric = -float('inf')
            self.best_model_metric = best_model_metric
            self.best_epoch = None

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            logging.info(f"### EPOCH: {int(state.epoch)} ###")
            # Note: "eval_" is prepended to the keys in metrics
            current_metric = metrics["eval_" + self.best_model_metric]
            pretty_metrics = json.dumps(metrics, indent=4)
            logging.info(pretty_metrics)
            if state.epoch % 5 == 0 or current_metric > self.best_metric:
                prompt = "frozen peas and carrots"
                test_inference(model, tokenizer, prompt, device)
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = state.epoch
                logging.info(f"Best model updated at epoch: {state.epoch} with metric ({current_metric})")

        def on_train_end(self, args, state, control, **kwargs):
            if self.best_epoch is not None:
                logging.info(f"The best model was saved from epoch: {self.best_epoch}")
                logging.info(f"The best result was {self.best_model_metric}: {self.best_metric}")

    # TODO: Move this to a separate file also
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

    CHECKPOINTS_DIR = Path(config['config']['checkpoints_dir'])
    RUN_PATH = CHECKPOINTS_DIR / run_name

    metric_for_best_model = config['training']['metric_for_best_model'] 
    
    training_args = TrainingArguments(
        output_dir=RUN_PATH,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        warmup_steps=100,
        max_grad_norm=1.0,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        report_to="wandb" if not SMOKE_TEST else None
    )

    # TODO: Add adam config (and scheduler?) to config file
    adamW = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-5, weight_decay=0.1)
    scheduler = CosineAnnealingWarmRestarts(adamW, T_0=2000, T_mult=1, eta_min=lr*0.1)
        
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(adamW, scheduler),  # Pass optimizers here (rather than training_args) for more fine-grained control
        callbacks=[SaveBestModelCallback(metric_for_best_model)]
    )

    logging.info("Training...")
    trainer.train()

    logging.info("Training complete")
    logging.info(f"Validation Results: {trainer.evaluate()}")

    # TODO: Include path
    logging.info("Saving the model...")
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    prompt = "frozen peas and carrots"
    test_inference(model, tokenizer, prompt, device)

    # TODO: Fix this for smoke_test 
    # » the logic of changing the output file is actually kinda tricky, need to go through inference setup
    FILENAME = "TestData_11.22.23.xlsx"
    # if SMOKE_TEST:
    #     FILENAME = "smoke_test_" + FILENAME
    INPUT_COLUMN = "Product Type"
    # TODO: Fix this...maybe need to update inference handler also
    DATA_DIR = "/net/projects/cgfp/data/raw/"

    INPUT_PATH = DATA_DIR + FILENAME

    output_sheet = inference_handler(
        model,
        tokenizer,
        input_path=INPUT_PATH,
        data_dir=DATA_DIR,
        device=device,
        sheet_name=0,
        input_column="Product Type",
        highlight=False,
        confidence_score=False,
        raw_results=False,
        assertion=False,
    )
    with pd.option_context('display.max_columns', None):
        logging.info(output_sheet.head(1))

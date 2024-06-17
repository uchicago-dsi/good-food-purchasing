import os
import logging
import json
from datetime import datetime
import yaml
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
import polars as pl
import numpy as np
import pandas as pd

from typing import Dict, Union, Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    TrainingArguments
)
from datasets import Dataset

import wandb

from cgfp.inference.inference import inference_handler, test_inference
from cgfp.training.models import MultiTaskConfig, MultiTaskModel
from cgfp.training.trainer import compute_metrics, SaveBestModelCallback, MultiTaskTrainer
from cgfp.constants.training_constants import LABELS, BASIC_TYPE_IDX, FPG_IDX

SCRIPT_DIR = Path(__file__).resolve().parent
device = "cuda:0" if torch.cuda.is_available() else "cpu"

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


if __name__ == "__main__":
    with open(SCRIPT_DIR / "config_train.yaml", "r") as file:
        config = yaml.safe_load(file)

    SMOKE_TEST = config['config']['smoke_test']

    # Data configuration
    TEXT_FIELD = config['data']['text_field']
    DATA_DIR = Path(config['data']['data_dir'])
    TRAIN_DATA_PATH = DATA_DIR / config['data']['train_filename']
    EVAL_DATA_PATH = DATA_DIR / config['data']['eval_filename']

    # Model configuration
    SAVE_BEST = config['model']['save_best']
    MODEL_NAME = config['model']['model_name']
    RESET_CLASSIFICATION_HEADS = config['model']['reset_classification_heads']
    ATTACHED_HEADS = config['model']['attached_heads']
    FREEZE_BASE = config['model']['freeze_base']

    checkpoint = config['model']['checkpoint']
    classification = config['model']['classification']
    loss = config['model']['loss']
    metric_for_best_model = config['training']['metric_for_best_model'] 

    # Training hyperparameters
    lr = config['training']['lr']
    epochs = config['training']['epochs'] if not SMOKE_TEST else 6
    train_batch_size = config['training']['train_batch_size']
    eval_batch_size = config['training']['eval_batch_size']

    eval_prompt = config['training']['eval_prompt']

    betas = tuple(config['adamw']['betas'])
    eps = float(config['adamw']['eps'])  # Note: scientific notation, so convert to float
    weight_decay = config['adamw']['weight_decay']

    T_0 = config['scheduler']['T_0']
    T_mult = config['scheduler']['T_mult']
    eta_min_constant = config['scheduler']['eta_min_constant']

    # Directory configuration
    RUN_NAME = f"{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    MODEL_SAVE_PATH = Path(config['model']['model_dir']) / RUN_NAME
    CHECKPOINTS_DIR = Path(config['config']['checkpoints_dir'])
    RUN_PATH = CHECKPOINTS_DIR / RUN_NAME

    # Logging configuration
    if SMOKE_TEST:
        RUN_NAME += "-smoke-test"
        os.environ["WANDB_DISABLED"] = "true"
    else:
        wandb.init(project='cgfp', name=RUN_NAME)

    LOGGING_DIR = SCRIPT_DIR.parent / "logs/"
    LOG_FILE = LOGGING_DIR / f"{RUN_NAME}.log"

    logging.basicConfig(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

    transformers_logger = logging.getLogger('transformers')
    transformers_logger.setLevel(logging.INFO)
    transformers_logger.addHandler(file_handler)

    ### SETUP ###
    logging.info(f"MODEL_PATH : {MODEL_SAVE_PATH}")
    logging.info("Starting...")
    logging.info(f"Training with device : {device}")
    logging.info(f"Using base model : {MODEL_NAME}")
    logging.info(f"Predicting based on input field : {TEXT_FIELD}")
    logging.info(f"Predicting categorical fields : {LABELS}")

    ### DATA PREP ###
    logging.info(f"Reading training data from path : {TRAIN_DATA_PATH}")
    df_train = read_data(TEXT_FIELD, LABELS, TRAIN_DATA_PATH)

    logging.info(f"Reading eval data from path : {TRAIN_DATA_PATH}")
    df_eval = read_data(TEXT_FIELD, LABELS, EVAL_DATA_PATH)

    df_combined = pl.concat([df_train, df_eval]) # combine training and eval so we have all valid outputs for evaluation

    # TODO: Put this in a function
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

    # TODO: Put tokenizing the dataset in a function
    train_dataset = train_dataset.map(tokenize)
    eval_dataset = eval_dataset.map(tokenize)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    logging.info("Datasets are prepared")
    logging.info(f"Structure of the dataset : {train_dataset}")
    logging.info(f"Sample record from the dataset : {train_dataset[0]}")

    ### MODEL SETUP ###
    logging.info("Instantiating model")

    # TODO: Stop using this now that we have the counts. Use counts object instead in models.py
    num_categories_per_task = [len(v.classes_) for k, v in encoders.items()]

    if checkpoint is None:
        # If no specified checkpoint, use pretrained Huggingface model
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

    if ATTACHED_HEADS is not None:
        model.set_attached_heads(ATTACHED_HEADS)
    else:
        model.set_attached_heads(LABELS)

    if FREEZE_BASE:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the classification heads
        for name, head in model.classification_heads.items():
            for param in head.parameters():
                param.requires_grad = True

    logging.info("Model instantiated")
    
    ### TRAINING ###
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

    adamW = AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(adamW, T_0=T_0, T_mult=T_mult, eta_min=lr*eta_min_constant)
        
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(adamW, scheduler),  # Pass optimizers here (rather than training_args) for more fine-grained control
        callbacks=[SaveBestModelCallback(model, tokenizer, device, metric_for_best_model, eval_prompt)]
    )

    logging.info("Training...")
    trainer.train()

    logging.info("Training complete")
    logging.info(f"Validation Results: {trainer.evaluate()}")
    logging.info(f"Saving the model to {MODEL_SAVE_PATH}...")
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    ### EVAL & MODEL SAVING ###
    eval_prompt = "frozen peas and carrots"
    test_inference(model, tokenizer, eval_prompt, device)

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

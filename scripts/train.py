"""Trains multitask classification model for the Center for Good Food Purchasing"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    TrainingArguments,
)

from cgfp.constants.training_constants import LABELS
from cgfp.inference.inference import inference_handler, test_inference
from cgfp.training.models import MultiTaskConfig, MultiTaskModel
from cgfp.training.trainer import MultiTaskTrainer, SaveBestModelCallback, compute_metrics

SCRIPT_DIR = Path(__file__).resolve().parent
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_paths(directory: Path, filenames: Union[str, List[str]]) -> List[Path]:
    """Generates a list of file paths by combining directory and filenames.

    This function handles both single filenames and lists of filenames. Used to handle filepaths from config

    Args:
        directory: The base directory where the files are located.
        filenames: A single filename or a list of filenames to be combined with the directory.

    Returns:
        A list of Path objects corresponding to the provided filenames.
    """
    if isinstance(filenames, list):
        return [directory / filename for filename in filenames]
    else:
        return [directory / filenames]


def read_data(input_col: str, labels: list[str], data_path: str, smoke_test: bool = False) -> pd.DataFrame:
    """Reads data from a CSV file, filters out rows with null values in specified columns, and returns a cleaned DataFrame with specified columns.

    Args:
        input_col: The name of the input column that will be used for predictions
        labels: A list of column names to be included in the returned DataFrame.
        data_path: The path to the CSV file containing the data.
        smoke_test: If True, limits the number of rows to read for testing purposes.

    Returns:
        A cleaned DataFrame containing only the specified columns.
    """
    logging.info(f"Reading data from {data_path}")
    nrows = 1000 if smoke_test else None

    file_extension = Path(data_path).suffix.lower()
    if file_extension == ".csv":
        df_cgfp = pd.read_csv(data_path, na_values=["NULL"], nrows=nrows)
    elif file_extension in [".xls", ".xlsx"]:
        df_cgfp = pd.read_excel(data_path, na_values=["NULL"], nrows=nrows)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    # Filter out rows with null values in specific columns
    for col in [input_col, "Food Product Group", "Food Product Category", "Primary Food Product Category"]:
        prev_height = df_cgfp.shape[0]
        df_cgfp = df_cgfp[df_cgfp[col].notna()]
        new_height = df_cgfp.shape[0]
        logging.info(f"Excluded {prev_height - new_height} rows due to null values in '{col}'.")

    keep_cols = [input_col] + labels
    df_cleaned = df_cgfp[keep_cols]

    # Make sure input_col is correctly encoded as string
    # TODO: fix slice warning here
    df_cleaned[input_col] = df_cleaned[input_col].astype(str)

    df_cleaned = df_cleaned.fillna("None")

    return df_cleaned


def get_encoder(unique_categories: list[str]) -> LabelEncoder:
    """Initializes and fits a LabelEncoder on the provided unique categories.

    Args:
        unique_categories: A list of unique categories to fit the encoder.

    Returns:
        A fitted LabelEncoder instance.
    """
    encoder = LabelEncoder()
    encoder.fit(unique_categories)
    return encoder


def get_encoders_and_counts(
    df: pd.DataFrame, labels: list[str] = LABELS
) -> tuple[dict[str, LabelEncoder], dict[str, dict]]:
    """Generates LabelEncoders and counts for specified label columns in a DataFrame.

    Args:
        df: The DataFrame containing the data.
        labels: A list of label columns to generate encoders and counts for.

    Returns:
        A tuple containing:
        - A dictionary of LabelEncoders for the specified label columns.
        - A dictionary of value counts for the specified label columns.
    """
    encoders = {}
    counts = {}
    subtype_counts = pd.Series(dtype=int)

    for column in labels:
        col_counts = df[column].value_counts().sort_index()
        logging.info(f"Categories for {column}")
        logging.info(col_counts)
        counts[column] = col_counts.to_dict()
        if "Sub-Type" in column:
            # Aggregate all sub-type options and handle at the end
            # Note: We are maintaining individual Sub-Type and collective Sub-Type encoders
            # so we can sort sub-types at inference time
            col_counts = df[column].value_counts().sort_index()
            # Note: This handles duplicates correctly
            subtype_counts = subtype_counts.add(col_counts, fill_value=0).astype(int)
        else:
            # Note: We want counts for each sub-type, but not an encoder
            encoders[column] = get_encoder(col_counts.index)

    # Handle sub-types
    subtype_counts = subtype_counts.sort_index()
    logging.info("Categories for Sub-Types")
    logging.info(subtype_counts)
    counts["Sub-Types"] = subtype_counts.to_dict()
    encoders["Sub-Types"] = get_encoder(subtype_counts.index)

    return encoders, counts


def get_decoders(encoders: dict[str, LabelEncoder]) -> list[tuple[str, dict[str, str]]]:
    """Generates decoders from the provided encoders for saving to a model config.

    Args:
        encoders: A dictionary of LabelEncoders for different columns.

    Returns:
        A list of tuples where each tuple contains a column name and its corresponding decoding dictionary.
    """
    decoders = []
    for col, encoder in encoders.items():
        # Note: Huggingface is picky with config.json...a list of tuples works
        decoding_dict = {f"{index}": label for index, label in enumerate(encoder.classes_)}
        decoders.append((col, decoding_dict))
    return decoders


def get_inference_masks(df_cgfp: pd.DataFrame, encoders: dict[str, LabelEncoder]) -> dict[str, list[int]]:
    """Generates inference masks for valid basic types within each food product group.

    Args:
        df_cgfp: The DataFrame containing the CGFP input data.
        encoders: A dictionary of LabelEncoders for different columns.

    Returns:
        A dictionary where each key is a food product group, and the value is a list representing
        a mask of valid basic types.
    """
    inference_masks = {}
    basic_types = df_cgfp["Basic Type"].unique().tolist()
    food_product_groups = df_cgfp["Food Product Group"].unique().tolist()
    for fpg in food_product_groups:
        valid_basic_types = df_cgfp[df_cgfp["Food Product Group"] == fpg]["Basic Type"].unique().tolist()
        basic_type_indices = encoders["Basic Type"].transform(valid_basic_types)
        mask = np.zeros(len(basic_types))
        mask[basic_type_indices] = 1
        inference_masks[fpg] = mask.tolist()
    return inference_masks


def tokenize(example: dict[str, str], labels: list[str] = LABELS) -> dict[str, list]:
    """Tokenizes the input text and transforms the specified labels using encoders.

    Args:
        example: A dictionary containing the text and label data for a single example.
        labels: A list of label columns to be tokenized and encoded.

    Returns:
        A dictionary containing the tokenized inputs and the corresponding encoded labels.
    """
    # Note: lowercase text since not all models are uncased and text is usually all caps
    tokenized_inputs = tokenizer(example[TEXT_FIELD].lower(), padding="max_length", truncation=True, max_length=100)
    tokenized_labels = []
    for label in LABELS:
        if "Sub-Type" in label:
            tokenized_labels.append(encoders["Sub-Types"].transform([example[label]]))
        else:
            tokenized_labels.append(encoders[label].transform([example[label]]))
    tokenized_inputs["labels"] = tokenized_labels
    return tokenized_inputs


if __name__ == "__main__":
    with Path.open(SCRIPT_DIR / "config_train.yaml") as file:
        config = yaml.safe_load(file)

    SMOKE_TEST = config["config"]["smoke_test"]

    # Data & directory configuration
    DATA_DIR = Path(config["data"]["data_dir"])
    CLEAN_DIR = DATA_DIR / "clean"
    RAW_DIR = DATA_DIR / "raw"
    TEST_DIR = DATA_DIR / "test"
    Path.mkdir(CLEAN_DIR, exist_ok=True, parents=True)
    Path.mkdir(RAW_DIR, exist_ok=True, parents=True)
    Path.mkdir(TEST_DIR, exist_ok=True, parents=True)

    train_data_paths = get_paths(CLEAN_DIR, config["data"]["train_filenames"])
    eval_data_paths = get_paths(CLEAN_DIR, config["data"]["eval_filenames"])

    TEST_DATA_PATH = RAW_DIR / config["data"]["test_filename"]

    TEXT_FIELD = config["data"]["text_field"]

    # Model configuration
    SAVE_BEST = config["model"]["save_best"]
    MODEL_NAME = config["model"]["model_name"]
    RESET_CLASSIFICATION_HEADS = config["model"]["reset_classification_heads"]
    ATTACHED_HEADS = config["model"]["attached_heads"]
    FREEZE_BASE = config["model"]["freeze_base"]
    UPDATE_CONFIG = config["model"]["update_config"]

    pretrained_checkpoint = config["model"]["starting_checkpoint"]
    if pretrained_checkpoint is None:
        if MODEL_NAME == "distilbert":
            starting_checkpoint = "distilbert-base-uncased"
        elif MODEL_NAME == "roberta":
            starting_checkpoint = "FacebookAI/roberta-base"
    else:
        starting_checkpoint = pretrained_checkpoint

    classification_head_type = config["model"]["classification_head_type"]
    loss = config["model"]["loss"]
    metric_for_best_model = config["training"]["metric_for_best_model"]

    # Training hyperparameters
    lr = float(config["training"]["lr"])
    epochs = config["training"]["epochs"] if not SMOKE_TEST else 6
    train_batch_size = config["training"]["train_batch_size"]
    eval_batch_size = config["training"]["eval_batch_size"]

    eval_prompt = config["training"]["eval_prompt"]

    betas = tuple(config["adamw"]["betas"])
    eps = float(config["adamw"]["eps"])  # Note: scientific notation, so convert to float
    weight_decay = config["adamw"]["weight_decay"]

    T_0 = config["scheduler"]["T_0"]
    T_mult = config["scheduler"]["T_mult"]
    eta_min_constant = config["scheduler"]["eta_min_constant"]

    # Directory configuration
    RUN_TITLE = config["config"]["run_title"]
    RUN_TITLE = RUN_TITLE.replace(" ", "_") if RUN_TITLE is not None else ""
    RUN_NAME = f"{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    RUN_NAME += "_" + RUN_TITLE
    if SMOKE_TEST:
        RUN_NAME += "_smoke_test"
    MODEL_SAVE_PATH = Path(config["model"]["model_dir"]) / RUN_NAME
    CHECKPOINTS_DIR = Path(config["config"]["checkpoints_dir"])
    RUN_PATH = CHECKPOINTS_DIR / RUN_NAME
    LOGGING_DIR = SCRIPT_DIR.parent / "logs/"

    # Logging configuration
    LOG_FILE = LOGGING_DIR / f"{RUN_NAME}.log"

    if SMOKE_TEST:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        wandb.init(project="cgfp", name=RUN_NAME)

    # TODO: This still behaves oddly with slurm, etc...
    logging.basicConfig(level=logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

    transformers_logger = logging.getLogger("transformers")
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
    logging.info(f"Reading training data from path : {train_data_paths}")
    df_train = pd.concat(
        [read_data(TEXT_FIELD, LABELS, train_path, smoke_test=SMOKE_TEST) for train_path in train_data_paths],
        ignore_index=True,
    )
    # Drop duplicate input column rows for more balanced dataset
    prev_height = df_train.shape[0]
    df_train = df_train.drop_duplicates(subset=[TEXT_FIELD])
    new_height = df_train.shape[0]
    logging.info(f"Dropped {prev_height - new_height} duplicate rows based on the column '{TEXT_FIELD}'.")

    logging.info(f"Reading eval data from path : {train_data_paths}")
    df_eval = pd.concat(
        [read_data(TEXT_FIELD, LABELS, eval_path, smoke_test=SMOKE_TEST) for eval_path in eval_data_paths],
        ignore_index=True,
    )

    prev_height = df_eval.shape[0]
    df_eval = df_eval.drop_duplicates(subset=[TEXT_FIELD])
    new_height = df_train.shape[0]
    logging.info(f"Dropped {prev_height - new_height} duplicate rows based on the column '{TEXT_FIELD}'.")

    labels_dict = {label: i for i, label in enumerate(LABELS)}

    df_combined = pd.concat(
        [df_train, df_eval]
    )  # combine training and eval so we have all valid outputs for evaluation

    encoders, counts = get_encoders_and_counts(df_combined)
    decoders = get_decoders(encoders)
    inference_masks = get_inference_masks(df_combined, encoders)

    logging.info("Preparing dataset")
    if MODEL_NAME == "distilbert":
        tokenizer = DistilBertTokenizerFast.from_pretrained(starting_checkpoint)
    elif MODEL_NAME == "roberta":
        tokenizer = RobertaTokenizerFast.from_pretrained(starting_checkpoint)

    train_dataset = Dataset.from_pandas(df_train)
    eval_dataset = Dataset.from_pandas(df_eval)

    train_dataset = train_dataset.map(tokenize)
    eval_dataset = eval_dataset.map(tokenize)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    logging.info("Datasets are prepared")
    logging.info(f"Structure of the dataset : {train_dataset}")
    logging.info(f"Sample record from the dataset : {train_dataset[0]}")

    ### MODEL SETUP ###
    logging.info("Instantiating model")

    if pretrained_checkpoint is None:
        # If no specified checkpoint, use pretrained Huggingface model
        logging.info(f"Loading model from the off-the-shelf Huggingface {starting_checkpoint}")
        if MODEL_NAME == "distilbert":
            base_model = DistilBertForSequenceClassification.from_pretrained(starting_checkpoint)
        elif MODEL_NAME == "roberta":
            base_model = RobertaForSequenceClassification.from_pretrained(starting_checkpoint)
            base_model.config.classifier_dropout = 0.2  # TODO: put this in config

        base_model_config = base_model.config.to_dict()
        if "model_type" not in base_model_config:
            base_model_config["model_type"] = MODEL_NAME

        multi_task_config = MultiTaskConfig(
            decoders=decoders,
            columns=labels_dict,
            classification=classification_head_type,
            inference_masks=json.dumps(inference_masks),
            counts=json.dumps(counts),
            loss=loss,
            **base_model_config,
        )
        model = MultiTaskModel(multi_task_config)
    else:
        logging.info(f"Loading model from {starting_checkpoint}")
        multi_task_config = MultiTaskConfig.from_pretrained(starting_checkpoint)

        # Note: Smoke test will overwrite model config with limited data
        if UPDATE_CONFIG:
            multi_task_config.decoders = decoders
            multi_task_config.columns = labels_dict
            multi_task_config.classification = classification_head_type
            multi_task_config.inference_masks = json.dumps(inference_masks)
            multi_task_config.counts = json.dumps(counts)
            multi_task_config.loss = loss
            multi_task_config.base_model_type = MODEL_NAME

        # Note: ignore_mismatched_sizes since we are often loading from checkpoints with different numbers of categories
        model = MultiTaskModel.from_pretrained(
            starting_checkpoint, config=multi_task_config, ignore_mismatched_sizes=True
        )

        model.initialize_inference_masks()
        model.initialize_tasks()
        model.initialize_losses()

    if RESET_CLASSIFICATION_HEADS:
        logging.info("Resetting classification heads...")
        model.initialize_classification_heads()

    if ATTACHED_HEADS is not None:
        model.set_attached_heads(ATTACHED_HEADS)
    else:
        classification_head_labels = [label for label in LABELS if "Sub-Type" not in label]
        classification_head_labels += ["Sub-Types"]
        model.set_attached_heads(classification_head_labels)

    if FREEZE_BASE:
        logging.info("Freezing base model...")

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the classification heads
        for head in model.classification_heads.values():
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
        report_to="wandb" if not SMOKE_TEST else None,
    )

    adamW = AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(adamW, T_0=T_0, T_mult=T_mult, eta_min=lr * eta_min_constant)

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        compute_metrics=lambda p: compute_metrics(
            p, model
        ),  #  Note: We need to access model config during compute metrics
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(
            adamW,
            scheduler,
        ),  #  Pass optimizers here (rather than training_args) for more fine-grained control
        callbacks=[SaveBestModelCallback(model, tokenizer, device, metric_for_best_model, eval_prompt)],
    )

    logging.info("Evaluating model before training...")
    pre_train_metrics = trainer.evaluate()
    logging.info("Pre-training evaluation metrics:", pre_train_metrics)

    logging.info("Training...")
    trainer.train()

    logging.info("Training complete")
    logging.info(f"Validation Results: {trainer.evaluate()}")
    logging.info(f"Saving the model to {MODEL_SAVE_PATH}...")
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    ### EVAL & MODEL SAVING ###
    test_inference(model, tokenizer, eval_prompt, device)

    output_sheet = inference_handler(
        model,
        tokenizer,
        input_path=TEST_DATA_PATH,
        save_dir=DATA_DIR,
        device=device,
        sheet_name=0,
        input_column=TEXT_FIELD,
        confidence_score=False,
        raw_results=False,
        assertion=False,
        save=not SMOKE_TEST,
    )
    with pd.option_context("display.max_columns", None):
        logging.info(output_sheet.head(1))

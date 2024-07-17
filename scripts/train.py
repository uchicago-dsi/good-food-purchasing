import ast
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
import yaml
from cgfp.constants.training_constants import LABELS
from cgfp.inference.inference import inference_handler, test_inference
from cgfp.training.models import MultiTaskConfig, MultiTaskModel
from cgfp.training.trainer import MultiTaskTrainer, SaveBestModelCallback, compute_metrics
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, TrainingArguments

import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def read_data_polars(input_col, labels, data_path):
    # Note: polars syntax is different than pandas syntax
    df = pl.read_csv(data_path, infer_schema_length=1, null_values=["NULL"]).lazy()
    for col in [input_col, "Food Product Group", "Food Product Category", "Primary Food Product Category"]:
        prev_height = df.collect().shape[0]
        df = df.filter(pl.col(col).is_not_null())
        new_height = df.collect().shape[0]
        logging.info(f"Excluded {prev_height - new_height} rows due to null values in '{col}'.")

    # TODO: I think this isn't a good idea...should just set this up so that the data comes in with a list of Sub-Types
    # # Handle "Sub-Types" — may not always be in dataset
    # if "Sub-Types" not in df.collect().columns:
    #     sub_type_cols = [col for col in df.columns if "Sub-Type" in col]
    #     if sub_type_cols:
    #         df = df.with_columns(
    #             pl.concat_list(sub_type_cols).alias("Sub-Types")
    #         )

    df_cleaned = df.select(input_col, *labels)

    # Convert 'Sub-Types' from string back to list
    def str_to_list(s):
        return ast.literal_eval(s) if s else []

    df_cleaned = df_cleaned.with_columns(pl.col("Sub-Types").apply(str_to_list))

    # Make sure every input_col is correctly encoded as string
    df_cleaned = df_cleaned.with_columns(pl.col(input_col).cast(pl.Utf8))

    # TODO: should we use FPC to fix nulls for PFPC? This is a pipeline question
    df_cleaned = df_cleaned.fill_null("None")
    return df_cleaned


def read_data(input_col, labels, data_path, smoke_test=False, combine_subtypes=False):
    nrows = 1000 if smoke_test else None
    df_cgfp = pd.read_csv(data_path, na_values=["NULL"], nrows=nrows)

    # Filter out rows with null values in specific columns
    for col in [input_col, "Food Product Group", "Food Product Category", "Primary Food Product Category"]:
        prev_height = df_cgfp.shape[0]
        df_cgfp = df_cgfp[df_cgfp[col].notna()]
        new_height = df_cgfp.shape[0]
        logging.info(f"Excluded {prev_height - new_height} rows due to null values in '{col}'.")

    keep_cols = [input_col] + labels

    # TODO: This can probably go away...handling sub-types in the forward pass
    # def combine_sub_types(row):
    #     return [item for item in row if pd.notna(item) and item is not None]

    # if combine_subtypes:
    #     df_cgfp["Sub-Types"] = df_cgfp[subtype_cols].apply(combine_sub_types, axis=1)
    #     keep_cols.append("Sub-Types")

    df_cleaned = df_cgfp[keep_cols]

    # Make sure input_col is correctly encoded as string
    # TODO: fix slice warning here
    df_cleaned[input_col] = df_cleaned[input_col].astype(str)

    df_cleaned = df_cleaned.fillna("None")

    return df_cleaned


def get_encoder(unique_categories):
    encoder = LabelEncoder()
    encoder.fit(unique_categories)
    return encoder


def get_encoders_and_counts(df, labels=LABELS):
    encoders = {}
    counts = {}
    subtype_counts = pd.Series(dtype=int)

    for column in labels:
        col_counts = df[column].value_counts().sort_index()
        logging.info(f"Categories for {column}")
        logging.info(col_counts)
        counts[column] = col_counts.to_list()
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
    counts["Sub-Types"] = subtype_counts.to_list()
    encoders["Sub-Types"] = get_encoder(subtype_counts.index)

    return encoders, counts


def get_decoders(encoders):
    # Create decoders to save to model config
    decoders = []
    for col, encoder in encoders.items():
        # Note: Huggingface is picky with config.json...a list of tuples works
        decoding_dict = {f"{index}": label for index, label in enumerate(encoder.classes_)}
        decoders.append((col, decoding_dict))
    return decoders


def get_inference_masks_polars(df, encoders):
    # Save valid basic types for each food product group
    inference_masks = {}
    basic_types = df.select("Basic Type").unique().collect()["Basic Type"].to_list()
    for fpg in df.select("Food Product Group").unique().collect()["Food Product Group"]:
        # Note: polars syntax is different than pandas
        valid_basic_types = (
            df.filter(pl.col("Food Product Group") == fpg)
            .select("Basic Type")
            .unique()
            .collect()["Basic Type"]
            .to_list()
        )
        basic_type_indeces = encoders["Basic Type"].transform(valid_basic_types)
        mask = np.zeros(len(basic_types))
        mask[basic_type_indeces] = 1
        inference_masks[fpg] = mask.tolist()
    return inference_masks


def get_inference_masks(df, encoders):
    """Save valid basic types for each food product group"""
    inference_masks = {}
    basic_types = df["Basic Type"].unique().tolist()
    food_product_groups = df["Food Product Group"].unique().tolist()
    for fpg in food_product_groups:
        valid_basic_types = df[df["Food Product Group"] == fpg]["Basic Type"].unique().tolist()
        basic_type_indices = encoders["Basic Type"].transform(valid_basic_types)
        mask = np.zeros(len(basic_types))
        mask[basic_type_indices] = 1
        inference_masks[fpg] = mask.tolist()
    return inference_masks


def tokenize(example, labels=LABELS):
    tokenized_inputs = tokenizer(example[TEXT_FIELD], padding="max_length", truncation=True, max_length=100)
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

    # Data configuration
    DATA_DIR = Path(config["data"]["data_dir"])
    CLEAN_DIR = DATA_DIR / "clean"
    RAW_DIR = DATA_DIR / "raw"
    TEST_DIR = DATA_DIR / "test"
    Path.mkdir(CLEAN_DIR, exist_ok=True, parents=True)
    Path.mkdir(RAW_DIR, exist_ok=True, parents=True)
    Path.mkdir(TEST_DIR, exist_ok=True, parents=True)

    TRAIN_DATA_PATH = CLEAN_DIR / config["data"]["train_filename"]
    EVAL_DATA_PATH = CLEAN_DIR / config["data"]["eval_filename"]
    TEST_DATA_PATH = RAW_DIR / config["data"]["test_filename"]

    TEXT_FIELD = config["data"]["text_field"]

    # Model configuration
    SAVE_BEST = config["model"]["save_best"]
    MODEL_NAME = config["model"]["model_name"]
    RESET_CLASSIFICATION_HEADS = config["model"]["reset_classification_heads"]
    ATTACHED_HEADS = config["model"]["attached_heads"]
    FREEZE_BASE = config["model"]["freeze_base"]
    COMBINE_SUBTYPES = config["model"]["combine_subtypes"]

    starting_checkpoint = config["model"]["starting_checkpoint"]
    classification = config["model"]["classification"]
    loss = config["model"]["loss"]
    metric_for_best_model = config["training"]["metric_for_best_model"]

    # Training hyperparameters
    lr = config["training"]["lr"]
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
    RUN_NAME = f"{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    if SMOKE_TEST:
        RUN_NAME += "-smoke-test"
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
    logging.info(f"Reading training data from path : {TRAIN_DATA_PATH}")
    df_train = read_data(TEXT_FIELD, LABELS, TRAIN_DATA_PATH, combine_subtypes=COMBINE_SUBTYPES, smoke_test=SMOKE_TEST)

    logging.info(f"Reading eval data from path : {TRAIN_DATA_PATH}")
    # TODO: doing this as a smoke test...
    EVAL_DATA_PATH = TRAIN_DATA_PATH
    df_eval = read_data(TEXT_FIELD, LABELS, EVAL_DATA_PATH, combine_subtypes=COMBINE_SUBTYPES, smoke_test=SMOKE_TEST)

    labels_dict = {label: i for i, label in enumerate(LABELS)}

    df_combined = pd.concat(
        [df_train, df_eval]
    )  # combine training and eval so we have all valid outputs for evaluation

    encoders, counts = get_encoders_and_counts(df_combined)
    decoders = get_decoders(encoders)
    inference_masks = get_inference_masks(df_combined, encoders)

    logging.info("Preparing dataset")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_dataset = Dataset.from_pandas(df_train)
    eval_dataset = Dataset.from_pandas(df_eval)

    train_dataset = train_dataset.map(tokenize)
    eval_dataset = eval_dataset.map(tokenize)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # TODO: Kinda hacky since we need to skip "Product Type" when counting our columns...
    # TODO: Do I actually need this?
    # Kind of...sort this out later for sub-type columns...
    # subtype_indices = [i - 1 for i, col in enumerate(train_dataset.features) if "Sub-Type" in col]

    logging.info("Datasets are prepared")
    logging.info(f"Structure of the dataset : {train_dataset}")
    logging.info(f"Sample record from the dataset : {train_dataset[0]}")

    ### MODEL SETUP ###
    logging.info("Instantiating model")

    if starting_checkpoint is None:
        # If no specified checkpoint, use pretrained Huggingface model
        distilbert_model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
        multi_task_config = MultiTaskConfig(
            decoders=decoders,
            columns=labels_dict,
            classification=classification,
            inference_masks=json.dumps(inference_masks),
            counts=json.dumps(counts),
            loss=loss,
            # subtype_indices=subtype_indices,
            **distilbert_model.config.to_dict(),
        )
        model = MultiTaskModel(multi_task_config)
    else:
        # Note: ignore_mismatched_sizes since we are often loading from checkpoints with different numbers of categories
        # model = MultiTaskModel.from_pretrained(starting_checkpoint, ignore_mismatched_sizes=True)

        # TODO: Kinda ugly...may also need to set something up so that we are not always overwriting config
        # TODO: This also breaks with smoke test...reinit the size of the categories somehow maybe?
        # config_dict = model.config.to_dict()
        multi_task_config = MultiTaskConfig.from_pretrained(starting_checkpoint)
        multi_task_config.decoders = decoders
        multi_task_config.columns = labels_dict
        multi_task_config.classification = classification
        multi_task_config.inference_masks = json.dumps(inference_masks)
        multi_task_config.counts = json.dumps(counts)
        multi_task_config.loss = loss
        # config_dict["decoders"] = decoders
        # config_dict["columns"] = labels_dict
        # config_dict["classification"] = classification
        # config_dict["inference_masks"] = json.dumps(inference_masks)
        # config_dict["counts"] = json.dumps(counts)
        # config_dict["loss"] = loss
        # TODO:
        # config_dict["subtype_indices"] = subtype_indices
        # multi_task_config = MultiTaskConfig(**config_updated)

        model = MultiTaskModel.from_pretrained(
            starting_checkpoint, config=multi_task_config, ignore_mismatched_sizes=True
        )

        # model.config = multi_task_config

        model.initialize_inference_masks()
        model.initialize_tasks()
        model.initialize_losses()

    if RESET_CLASSIFICATION_HEADS:
        model.initialize_classification_heads()

    if ATTACHED_HEADS is not None:
        model.set_attached_heads(ATTACHED_HEADS)
    else:
        # TODO: Handle sub-type configuration here
        classification_head_labels = [label for label in LABELS if "Sub-Type" not in label]
        classification_head_labels += ["Sub-Types"]
        model.set_attached_heads(classification_head_labels)

    if FREEZE_BASE:
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
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(
            adamW,
            scheduler,
        ),  # Pass optimizers here (rather than training_args) for more fine-grained control
        callbacks=[SaveBestModelCallback(model, tokenizer, device, metric_for_best_model, eval_prompt)],
    )

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
        highlight=False,
        confidence_score=False,
        raw_results=False,
        assertion=False,
        save=not SMOKE_TEST,
    )
    with pd.option_context("display.max_columns", None):
        logging.info(output_sheet.head(1))

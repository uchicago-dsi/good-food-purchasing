import os
import logging
import json
from datetime import datetime
import argparse

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import polars as pl
import numpy as np
import pandas as pd

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
    TrainerCallback
)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset

import wandb

from cgfp.inference.inference import inference, inference_handler
from cgfp.training.models import MultiTaskConfig, MultiTaskModel
from cgfp.config_training import LABELS

logging.basicConfig(level=logging.INFO)

def read_data(input_col, labels, data_path, drop_meals=False):
    # Note: polars syntax is different than pandas syntax
    df = pl.read_csv(data_path, infer_schema_length=1, null_values=["NULL"]).lazy()
    for col in [input_col, "Food Product Group", "Food Product Category", "Primary Food Product Category"]:
        prev_height = df.collect().shape[0]
        df = df.filter(pl.col(col).is_not_null())
        new_height = df.collect().shape[0]
        print(f"Excluded {prev_height - new_height} rows due to null values in '{col}'.")
    if drop_meals:
        df = df.filter(pl.col("Food Product Group") != "Meals")
    df_cleaned = df.select(input_col, *labels)
    # TODO: This maybe needs to be done to each column? 
    # Make sure every row is correctly encoded as string
    df_cleaned = df_cleaned.with_columns(pl.col(input_col).cast(pl.Utf8))
    # TODO: make this come from function args
    # TODO: should we use FPC to fix nulls for PFPC? This is a pipeline question
    df_cleaned = df_cleaned.fill_null("None")
    return df_cleaned

### DATA PREP ###

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
    parser = argparse.ArgumentParser()
    # Setup args
    parser.add_argument('--train_data_path', default="/net/projects/cgfp/data/clean/clean_CONFIDENTIAL_CGFP_bulk_data_073123.csv", type=str, help="Path to the training data CSV file.")
    parser.add_argument('--eval_data_path', default="/net/projects/cgfp/data/clean/combined_eval_set.csv", type=str, help="Path to the evaluation data CSV file.")
    # Config args
    parser.add_argument('--smoke_test', action='store_true', help="Run in smoke test mode to check basic functionality.")
    parser.add_argument('--keep_meals', action='store_true', help="Keep Meals items in training and eval datasets")
    parser.add_argument('--two_cols', action='store_true', help="Train only on food product group and basic type (for debugging)")
    # TODO: Is default behavior saving final model?
    parser.add_argument('--dont_save_best', action="store_false", help="Don't save the best model from the training run (saves the last model, I believe)")
    parser.add_argument('--train_attention', action="store_true", help="Trains all attention heads in the model (keeps MLPs frozen). (Default behavior is training only the classification heads)")
    parser.add_argument('--train_whole_model', action='store_true', help="Train the whole model. (Default behavior is training only the classification heads.)")
    parser.add_argument('--classification', default="mlp", type=str, help="Setup the classification heads. Choices: mlp, linear. Default is mlp")
    parser.add_argument('--loss', default="cross_entropy", type=str, help="Setup the loss function. Choices: cross_entropy, focal. Default is cross_entropy")
    # Hyperparameter args
    parser.add_argument('--lr', default=.001, type=float, help="Learning rate for the Huggingface Trainer")
    parser.add_argument('--epochs', default=30, type=int, help="Training epochs for the Huggingface Trainer")
    parser.add_argument('--train_batch_size', default=32, type=int, help="Training batch size for the Huggingface Trainer")
    parser.add_argument('--eval_batch_size', default=64, type=int, help="Evaluation batch size for the Huggingface Trainer")
    
    args = parser.parse_args()

    # Setup
    MODEL_NAME = "distilbert-base-uncased"
    TEXT_FIELD = "Product Type"

    # Config
    SMOKE_TEST = args.smoke_test
    SAVE_BEST = not args.dont_save_best
    DROP_MEALS = not args.keep_meals
    logging.info(f"DROP_MEALS: {DROP_MEALS}")
    FREEZE_MODEL = not args.train_whole_model
    FREEZE_MLPS = args.train_attention
    logging.info(f"FREEZE_MODEL: {FREEZE_MODEL}")
    logging.info(f"FREEZE_MLPS: {FREEZE_MLPS}")
    classification = args.classification
    loss = args.loss
    
    # Hyperparameters
    lr = args.lr
    epochs = 5 if SMOKE_TEST else args.epochs
    train_batch_size = args.train_batch_size # try 8,16,32
    eval_batch_size = args.eval_batch_size

    # Logging
    run_name = f"{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    if SMOKE_TEST:
        run_name += "-smoke-test"
        os.environ["WANDB_DISABLED"] = "true"
    MODEL_PATH = (
        f"/net/projects/cgfp/model-files/{run_name}"
    )
    if not SMOKE_TEST:
        wandb.init(project='cgfp', name=run_name)

    ### SETUP ###
    logging.info(f"MODEL_PATH : {MODEL_PATH}")
    logging.info("Starting")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info(f"Training with device : {device}")
    logging.info(f"Using base model : {MODEL_NAME}")
    logging.info(f"Predicting based on input field : {TEXT_FIELD}")
    logging.info(f"Predicting categorical fields : {LABELS}")

    ### DATA PREP ###
    if args.two_cols:
        LABELS = ["Food Product Group", "Basic Type"]
        logging.info("Training only on Food Product Group & Basic Type")
    # These indeces are used to set up inference filtering
    FPG_IDX = LABELS.index("Food Product Group")
    BASIC_TYPE_IDX = LABELS.index("Basic Type")
    logging.info(f"Reading training data from path : {args.train_data_path}")
    df_train = read_data(TEXT_FIELD, LABELS, args.train_data_path, drop_meals=DROP_MEALS)
    logging.info(f"Reading eval data from path : {args.train_data_path}")
    df_eval = read_data(TEXT_FIELD, LABELS, args.eval_data_path, drop_meals=DROP_MEALS)

    df_combined = pl.concat([df_train, df_eval]) # combine training and eval so we have all valid outputs for evaluation

    encoders = {}
    counts = {}
    for column in LABELS:
        # Create encoders (including all labels from training and eval sets)
        # Note: sort this so that the order is consistent
        unique_categories = df_combined.select(column).unique().sort(column).collect().to_numpy().ravel()
        encoder = LabelEncoder()
        encoder.fit(unique_categories)
        encoders[column] = encoder

        # Get counts for each category in the training set for focal loss
        counts_df = df_train.group_by(column).agg(pl.len().alias('count')).sort(column)

        # Fill 0s for categories that aren't in the training set
        full_counts_df = pl.DataFrame({column: unique_categories})
        full_counts_df = full_counts_df.join(counts_df.collect(), on=column, how='left').fill_null(0)
        counts[column] = full_counts_df['count'].to_list()

    # Create decoders to save to model config
    # Note: Huggingface is picky...so a list of tuples seems like the best bet
    # for saving to the config.json in a way that doesn't break when loaded
    decoders = []
    for col, encoder in encoders.items():
        decoding_dict = {
            f"{index}": label for index, label in enumerate(encoder.classes_)
        }
        decoders.append((col, decoding_dict))
        logging.info(f"{col}: {len(encoder.classes_)} classes")

    # Save valid basic types for each food product group
    # Note: polars syntax is different than pandas
    inference_masks = {}
    basic_types = df_combined.select("Basic Type").unique().collect()['Basic Type'].to_list()
    for fpg in df_combined.select('Food Product Group').unique().collect()['Food Product Group']:
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
    logging.info("Example data:")
    for i in range(5):
        logging.info(train_dataset[i])

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    logging.info("Dataset is prepared")
    logging.info(f"Structure of the dataset : {train_dataset}")
    logging.info(f"Sample record from the dataset : {train_dataset[0]}")

    ### TRAINING ###

    logging.info("Instantiating model")
    distilbert_model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased"
    )

    # TODO: Kill this now that we have the counts
    num_categories_per_task = [len(v.classes_) for k, v in encoders.items()]
    config = MultiTaskConfig(
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
    model = MultiTaskModel(config)
    logging.info("Model instantiated")

    # TODO: add an arg for freezing layers
    # Freeze all layers
    if FREEZE_MODEL:
        logging.info("Freezing model...")
        for param in model.parameters():
            param.requires_grad = False

        logging.info("Unfreezing classification heads...")
        # Unfreeze classification heads
        for param in model.classification_heads.parameters():
            param.requires_grad = True

    if FREEZE_MLPS:
        logging.info("Freezing model...")
        for param in model.parameters():
            param.requires_grad = False

        logging.info("Unfreezing attention and layernorn...")
        # Unfreeze attention heads and layernorm
        for name, param in model.named_parameters():
            if "attention" in name or "output" in name or "layer_norm" in name:
                param.requires_grad = True

        logging.info("Unfreezing classification heads...")
        # Unfreeze classification heads
        for param in model.classification_heads.parameters():
            param.requires_grad = True

        for name, param in model.named_parameters():
            print(f"{name} is {'frozen' if not param.requires_grad else 'unfrozen'}")

    class SaveBestModelCallback(TrainerCallback):
        def __init__(self, best_model_metric):
            self.best_metric = -float('inf')
            self.best_model_metric = best_model_metric
            self.best_epoch = None

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            # "eval_" is prepended to the keys in metrics
            current_metric = metrics["eval_" + self.best_model_metric]
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = state.epoch
                logging.info(f"Best model updated at epoch: {state.epoch} with metric ({current_metric})")

        def on_train_end(self, args, state, control, **kwargs):
            if self.best_epoch is not None:
                logging.info(f"The best model was saved from epoch: {self.best_epoch}")
                logging.info(f"The best result was {self.best_model_metric}: {self.best_metric}")

    # TODO: Training logs argument doesn't seem to work. Logs are in the normal logging folder?
    training_args = TrainingArguments(
        output_dir="/net/projects/cgfp/checkpoints",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        warmup_steps=100,
        logging_dir="./training-logs",
        max_grad_norm=1.0,
        report_to="wandb" if not SMOKE_TEST else None
    )

    best_model_metric = "basic_type_accuracy"
    if SAVE_BEST:
        training_args.load_best_model_at_end = True
        training_args.metric_for_best_model = best_model_metric # TODO: Or accuracy? Maybe should be basic type accuracy?
        training_args.greater_is_better = True

    # TODO: set this up to come from args
    adamW = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-5, weight_decay=0.1)
    scheduler = CosineAnnealingWarmRestarts(adamW, T_0=2000, T_mult=1, eta_min=lr*0.1)

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(adamW, scheduler), # Pass optimizers here (rather than training_args) for more fine-grained control
        callbacks=[SaveBestModelCallback(best_model_metric)]
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
    logging.info(inference(model, tokenizer, prompt, device, combine_name=True))

    # TODO: Fix this for smoke_test 
    # » the logic of changing the output file is actually kinda tricky, need to go through inference setup
    FILENAME = "TestData_11.22.23.xlsx"
    # if SMOKE_TEST:
    #     FILENAME = "smoke_test_" + FILENAME
    INPUT_COLUMN = "Product Type"
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
    pd.set_option('display.max_columns', None)
    logging.info(output_sheet.head(2))
    logging.info(f"Columns: {output_sheet.columns.tolist()}")

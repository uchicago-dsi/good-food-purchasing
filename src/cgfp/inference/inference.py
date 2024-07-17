import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from cgfp.constants.tokens.misc_tags import FPG2FPC
from cgfp.constants.training_constants import lower2label
from cgfp.training.models import MultiTaskModel
from torch.nn.functional import sigmoid
from transformers import DistilBertTokenizerFast

logger = logging.getLogger("inference_logger")
logger.setLevel(logging.INFO)


def test_inference(model, tokenizer, prompt, device="cuda:0"):
    normalized_name = inference(model, tokenizer, prompt, device, combine_name=True)
    logging.info(f"Example output for 'frozen peas and carrots': {normalized_name}")
    # TODO: I should set this up so I only do the forward pass once...
    preds_dict = inference(model, tokenizer, prompt, device)
    pretty_preds = json.dumps(preds_dict, indent=4)
    logging.info(pretty_preds)


def prediction_to_string(model, scores, idx):
    max_idx = torch.argmax(scores[idx])
    # decoders are a tuple of column name and actual decoding dictionary
    _, decoder = model.config.decoders[idx]
    return decoder[str(max_idx.item())]


def inference(
    model,
    tokenizer,
    text,
    device,
    assertion=False,
    confidence_score=True,
    combine_name=False,
):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    inputs = inputs.to(device)
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)

    # TODO: Need to pull out the subtype logits here
    # This is bad but I'm going to hardcode this for now
    nonsubtype_logits = outputs.logits[:-1]
    subtype_logits = outputs.logits[-1]

    # Note: Handle non-subtype logits first (multi-class classification)
    softmaxed_scores = [torch.softmax(logits, dim=1) for logits in nonsubtype_logits]

    fpg_idx = list(model.classification_heads.keys()).index("Food Product Group")
    fpc_idx = list(model.classification_heads.keys()).index("Food Product Category")

    # get predicted food product group & predicted food product category
    fpg = prediction_to_string(model, softmaxed_scores, fpg_idx)
    # TODO: This is fragile. Maybe change config to have a mapping of each column to index
    # TODO: This whole thing breaks if you have different columns that you're using...fix at some point
    fpc = prediction_to_string(model, softmaxed_scores, fpc_idx)

    inference_mask = model.inference_masks[fpg].to(device)

    # actually mask the basic type scores
    softmaxed_scores[model.config.columns["Basic Type"]] = (
        inference_mask * softmaxed_scores[model.config.columns["Basic Type"]]
    )

    # get the score for each task
    scores = [
        torch.max(score, dim=1) for score in softmaxed_scores
    ]  # Note: torch.max returns both max and argmax if you specify dim so this is a list of tuples

    decoders = {item[0]: item[1] for item in model.config.decoders}

    # assertion to make sure fpg & fpg match
    # TODO: Maybe good assertion behavior would be something like:
    # » If food product group + food product category + basic type don't match, ask GPT
    # » If one of these pairs doesn't match, just highlight
    # TODO: Add argument here to turn this behavior on and off
    assertion_failed = False
    if fpc not in FPG2FPC[fpg] and assertion:
        assertion_failed = True

    legible_preds = {}
    # TODO: This needs to be set up with the correct prediction head indeces
    for item, score in zip(decoders.items(), scores):
        col, decoder = item
        prob, idx = score

        try:
            pred = decoder[str(idx.item())]  # decoders have been serialized so keys are strings
            legible_preds[col] = pred if not assertion_failed else None
            if confidence_score:
                legible_preds[col + "_score"] = prob.item()
        except Exception:
            pass
            # TODO: what do we want to actually happen here?
            # Can we log or print base on where we are?
            # logging.info(f"Exception: {e}")

    # 4. Figure out what to do with the sub-types
    # - Idea: set up some sort of partial ordering and allow up to three sub-types if tokens
    # are in those sets
    # Need to set this up in a config somehwere
    threshold = 0.5
    # TODO: Also need to sort this to take the top three only...
    subtype_preds = (sigmoid(subtype_logits) > threshold).int()
    print(subtype_preds.sum())
    print(subtype_preds)
    subtype_indeces = torch.nonzero(subtype_preds.squeeze())
    print(subtype_indeces)
    legible_preds["Sub-Types"] = []
    for idx in subtype_indeces:
        legible_preds["Sub-Types"].append(decoders["Sub-Types"][str(idx.item())])
        # print(decoders["Sub-Types"][str(idx.item())])

    if combine_name:
        normalized_name = ""
        for col, pred in legible_preds.items():
            if "_score" not in col and "Food" not in col and pred != "None":
                normalized_name += pred + ", "
        normalized_name = normalized_name.strip().rstrip(",")
        return normalized_name
    return legible_preds


def highlight_uncertain_preds(df, threshold=0.85):
    """Creates a styles dictionary for underconfident predictions"""
    styles_dict = {}
    for col_idx, dtype in enumerate(df.dtypes):
        # TODO: this is fragile - fix it later
        if dtype == "object":  # Skip non-float columns
            continue
        else:
            try:
                styles_dict[df.columns[col_idx - 1]] = df.iloc[:, col_idx].apply(
                    lambda x: "background-color: yellow" if x < threshold else ""
                )
            except:
                print(f"Tried to find uncertainty in a a non-float column! {df.iloc[:,col_idx].head(5)}")
    return styles_dict


def save_output(df, filename, data_dir):
    os.chdir(data_dir)  # ensures this saves in the expected directory in Colab
    output_path = filename.rstrip(".xlsx") + "_classified.xlsx"
    df = df.replace("None", np.nan)
    df.to_excel(output_path, index=False)
    print(f"Classification completed! File saved to {output_path}")
    return


def inference_handler(
    model,
    tokenizer,
    input_path,
    input_column,
    output_filename=None,
    save_dir="/content",
    device=None,
    sheet_name=0,
    save=True,
    highlight=False,
    confidence_score=False,
    threshold=0.85,
    rows_to_classify=None,
    raw_results=False,
    assertion=False,
):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    try:
        df = pd.read_excel(input_path, sheet_name=sheet_name)
    except FileNotFoundError:
        print("FileNotFound: {e}\n. Please double check the filename: {input_path}")
        raise

    # Force columns to have capitalization consistent with expected output
    df.columns = [col.lower() for col in df.columns]
    df = df.rename(columns=lower2label)

    if rows_to_classify:
        df = df.head(rows_to_classify)

    output = (
        df[input_column]
        .apply(lambda text: inference(model, tokenizer, text, device, assertion=assertion))
        .apply(pd.Series)
    )
    results = pd.concat([df[input_column], output], axis=1)
    results = results.replace("None", pd.NA)

    if raw_results:
        if save:
            save_output(results, input_path, save_dir)
        return results

    # Add all columns to results to match name normalization format
    # Assumes that the input dataframe is in the expected name normalization format
    # TODO: Add a check for that
    results_full = pd.DataFrame()
    for col in df.columns:
        if col in results:
            results_full[col] = results[col]
        elif col == "Center Product ID":
            results_full[col] = df[col]
        else:
            results_full[col] = pd.Series([None] * len(results))

        # Add confidence score (needed for highlights)
        score_col = col + "_score"
        if score_col in results:
            results_full[score_col] = results[score_col]

    # Create highlights
    # Logic here is a bit odd since applying styles gives you a Styler
    # object...not a dataframe
    if highlight:
        styles_dict = highlight_uncertain_preds(results_full, threshold)
        styles_df = pd.DataFrame(styles_dict)

    if not confidence_score:
        results_full = results_full[[col for col in df.columns if "_score" not in col]]

    # Actually apply the styles here
    df_formatted = results_full.style.apply(lambda x: styles_df, axis=None) if highlight else results_full

    if output_filename is None:
        output_filename = input_path

    if save:
        save_output(df_formatted, output_filename, save_dir)
    return df_formatted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load model checkpoint.")
    parser.add_argument(
        "--filename",
        type=str,
        default="TestData_11.22.23.xlsx",
        help="Name of the file to classify.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the model checkpoint directory or Huggingface model name.",
    )

    args = parser.parse_args()
    checkpoint = args.checkpoint if args.checkpoint else "uchicago-dsi/cgfp-classifier-dev"
    FILENAME = args.filename

    model = MultiTaskModel.from_pretrained(checkpoint)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    SHEET_NUMBER = 0
    HIGHLIGHT = False
    CONFIDENCE_SCORE = False
    ROWS_TO_CLASSIFY = None
    RAW_RESULTS = False  # saves the raw model results rather than the formatted normalized name results
    ASSERTION = False

    # TODO: Put in constants or config file
    INPUT_COLUMN = "Product Type"
    DATA_DIR = "/net/projects/cgfp/data/raw/"
    INPUT_PATH = DATA_DIR + FILENAME

    inference_handler(
        model,
        tokenizer,
        input_path=INPUT_PATH,
        save_dir=DATA_DIR,
        device=device,
        sheet_name=SHEET_NUMBER,
        input_column=INPUT_COLUMN,
        rows_to_classify=ROWS_TO_CLASSIFY,
        highlight=HIGHLIGHT,
        confidence_score=CONFIDENCE_SCORE,
        raw_results=RAW_RESULTS,
        assertion=ASSERTION,
    )

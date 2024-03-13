import torch
import pandas as pd
import os
import logging
import numpy as np
import argparse

from cgfp.config_tags import GROUP_CATEGORY_VALIDATION
from cgfp.training.models import MultiTaskModel

from transformers import AutoTokenizer


logger = logging.getLogger("inference_logger")
logger.setLevel(logging.INFO)


def prediction_to_string(model, scores, idx):
    max_idx = torch.argmax(scores[idx])
    # model.decoders is a tuple of column name and actual decoding dictionary
    _, decoder = model.decoders[idx]
    return decoder[str(max_idx.item())]


def inference(model, tokenizer, text, device, assertion=True, confidence_score=True):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    inputs = inputs.to(device)
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
    softmaxed_scores = [torch.softmax(logits, dim=1) for logits in outputs.logits]

    # get predicted food product group & predicted food product category
    fpg = prediction_to_string(model, softmaxed_scores, model.fpg_idx)
    # TODO: This is fragile. Maybe change config to have a mapping of each column to index
    fpc = prediction_to_string(model, softmaxed_scores, model.fpg_idx + 1)

    inference_mask = model.inference_masks[fpg].to(device)

    # actually mask the basic type scores
    softmaxed_scores[model.basic_type_idx] = (
        inference_mask * softmaxed_scores[model.basic_type_idx]
    )

    # get the score for each task
    scores = [
        torch.max(score, dim=1) for score in softmaxed_scores
    ]  # torch.max returns both max and argmax if you specify dim so this is a list of tuples

    # 4. Figure out what to do with the sub-types
    # - Idea: set up some sort of partial ordering and allow up to three sub-types if tokens
    # are in those sets

    # assertion to make sure fpg & fpg match
    # TODO: Maybe good assertion behavior would be something like:
    # » If food product group + food product category + basic type don't match, ask GPT
    # » If one of these pairs doesn't match, just highlight
    # TODO: Add argument here to turn this behavior on and off
    assertion_failed = False
    if fpc not in GROUP_CATEGORY_VALIDATION[fpg] and assertion:
        assertion_failed = True

    legible_preds = {}
    for item, score in zip(model.decoders, scores):
        col, decoder = item
        prob, idx = score

        try:
            pred = decoder[
                str(idx.item())
            ]  # decoders have been serialized so keys are strings
            legible_preds[col] = pred if not assertion_failed else None
            if confidence_score:
                legible_preds[col + "_score"] = prob.item()
        except Exception as e:
            pass
            # TODO: what do we want to actually happen here?
            # Can we log or print base on where we are?
            # logging.info(f"Exception: {e}")
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
                print(
                    f"Tried to find uncertainty in a a non-float column! {df.iloc[:,col_idx].head(5)}"
                )
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
    data_dir="/content",
    device=None,
    sheet_name=0,
    highlight=False,
    confidence_score=False,
    threshold=0.85,
    rows_to_classify=None,
    raw_results=False,
    assertion=True,
):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    try:
        df = pd.read_excel(input_path, sheet_name=sheet_name)
    except FileNotFoundError as e:
        print("FileNotFound: {e}\n. Please double check the filename: {input_path}")
        raise

    if rows_to_classify:
        df = df.head(rows_to_classify)

    output = (
        df[input_column]
        .apply(
            lambda text: inference(model, tokenizer, text, device, assertion=assertion)
        )
        .apply(pd.Series)
    )
    results = pd.concat([df[input_column], output], axis=1)
    results = results.replace("None", pd.NA)

    if raw_results:
        save_output(results, input_path, data_dir)
        return

    # Add all columns to results to match name normalization format
    # Assumes that the input dataframe is in the expected name normalization format
    # TODO: Add a check for that
    # TODO: Seems like sub-types get dropped here?
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
    df_formatted = (
        results_full.style.apply(lambda x: styles_df, axis=None)
        if highlight
        else results_full
    )

    save_output(df_formatted, input_path, data_dir)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load model checkpoint.")
    parser.add_argument("--checkpoint", type=str, help="Path to the model checkpoint directory or Huggingface model name.")
    
    args = parser.parse_args()
    checkpoint = args.checkpoint if args.checkpoint else "uchicago-dsi/cgfp-classifier-dev"

    model = MultiTaskModel.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    SHEET_NUMBER = 0
    HIGHLIGHT = False
    CONFIDENCE_SCORE = False
    ROWS_TO_CLASSIFY = None
    RAW_RESULTS = False  # saves the raw model results rather than the formatted normalized name results
    ASSERTION = True

    FILENAME = "TestData_11.22.23.xlsx"
    INPUT_COLUMN = "Product Type"
    DATA_DIR = "/net/projects/cgfp/data/clean/"

    INPUT_PATH = DATA_DIR + FILENAME

    inference_handler(
        model,
        tokenizer,
        input_path=INPUT_PATH,
        data_dir=DATA_DIR,
        device=device,
        sheet_name=SHEET_NUMBER,
        input_column=INPUT_COLUMN,
        rows_to_classify=ROWS_TO_CLASSIFY,
        highlight=HIGHLIGHT,
        confidence_score=CONFIDENCE_SCORE,
        raw_results=RAW_RESULTS,
        assertion=ASSERTION,
    )

"""Contains inference functions for converting multitask model predictions to the expected output format for CGFP"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn.functional import sigmoid
from transformers import DistilBertTokenizerFast

from cgfp.constants.tokens.misc_tags import ALL_NON_SUBTYPE_TAGS, FPG2FPC
from cgfp.constants.training_constants import NON_LABEL_COLS, OUTPUT_COLS, lower2label
from cgfp.training.models import MultiTaskModel

logger = logging.getLogger("inference_logger")
logger.setLevel(logging.INFO)


def get_combined_name(legible_preds: dict[str, str]) -> str:
    """Combines predictions into a single, normalized name string.

    Args:
        legible_preds: A dictionary of predictions where keys are column names and values are the predicted strings.

    Returns:
        A combined, comma-separated string of the predicted normalized name
    """
    normalized_name = ""
    # TODO: Reorder this to be in the expected column output order
    for col, pred in legible_preds.items():
        if "_score" not in col and "Food" not in col and pred != "None" and pred is not None:
            normalized_name += pred + ", "
    normalized_name = normalized_name.strip().rstrip(",")
    return normalized_name


def test_inference(model: Any, tokenizer: Any, prompt: str, device: str = "cuda:0") -> None:
    """Performs inference on a given prompt using the specified model and tokenizer, logging the results.

    Args:
        model: The model used for inference
        tokenizer: The tokenizer used to preprocess the prompt
        prompt: The text input to be processed by the model
        device: The device on which to run the inference

    Returns:
        None
    """
    preds_dict = inference(model, tokenizer, prompt, device)
    normalized_name = get_combined_name(preds_dict)
    logging.info(f"Example output for 'frozen peas and carrots': {normalized_name}")
    pretty_preds = json.dumps(preds_dict, indent=4)
    logging.info(pretty_preds)


def prediction_to_string(model: Any, scores: torch.Tensor, idx: int) -> str:
    """Converts a model's prediction scores to a string representation using the appropriate decoder.

    Args:
        model: The model containing the configuration and decoders.
        scores: A tensor of prediction scores.
        idx: The index of the prediction to convert.

    Returns:
        The string representation of the prediction.
    """
    max_idx = torch.argmax(scores[idx])
    # Note: decoders are a tuple of column name and actual decoding dictionary
    _, decoder = model.config.decoders[idx]
    return decoder[str(max_idx.item())]


def inference(
    model: Any,
    tokenizer: Any,
    text: str,
    device: str,
    threshold: float = 0.5,
    assertion: bool = False,
    confidence_score: bool = False,
    combine_name: bool = False,
) -> Union[str, dict[str, Any]]:
    """Performs inference on the provided text using the specified model and tokenizer, returning either a combined name or a dictionary of predictions.

    Args:
        model: The model used for inference.
        tokenizer: The tokenizer used to preprocess the text input
        text: The input text to be processed by the model
        device: The device on which to run the inference
        threshold: The probability threshold to count as a positive prediction for multilabel tasks
        assertion: If True, performs an additional assertion check on the predictions
        confidence_score: If True, includes confidence scores in the predictions
        combine_name: If True, returns a combined string representation of the predictions

    Returns:
        A combined string of predictions if combine_name is True, otherwise a dictionary of predictions.
    """
    inputs = tokenizer(text.lower(), padding=True, truncation=True, return_tensors="pt")

    inputs = inputs.to(device)
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)

    # Pull out the sub-types logits to handle separately
    subtypes_idx = list(model.classification_heads.keys()).index("Sub-Types")
    nonsubtype_logits = outputs.logits[:subtypes_idx] + outputs.logits[subtypes_idx + 1 :]
    subtype_logits = outputs.logits[subtypes_idx]

    # Note: Handle non-subtype logits first (multi-class classification)
    softmaxed_scores = [torch.softmax(logits, dim=1) for logits in nonsubtype_logits]

    fpg_idx = list(model.classification_heads.keys()).index("Food Product Group")
    fpc_idx = list(model.classification_heads.keys()).index("Food Product Category")
    pfpc_idx = list(model.classification_heads.keys()).index("Primary Food Product Category")

    # Get predicted food product group & predicted food product category
    fpg = prediction_to_string(model, softmaxed_scores, fpg_idx)
    fpc = prediction_to_string(model, softmaxed_scores, fpc_idx)
    pfpc = prediction_to_string(model, softmaxed_scores, pfpc_idx)

    inference_mask = model.inference_masks[fpg].to(device)

    # actually mask the basic type scores
    softmaxed_scores[model.config.columns["Basic Type"]] = (
        inference_mask * softmaxed_scores[model.config.columns["Basic Type"]]
    )

    # get the score for each task
    scores = [
        torch.max(score, dim=1) for score in softmaxed_scores
    ]  # Note: torch.max returns both max and argmax if you specify dim so this is a list of tuples

    # Assertions to make sure that high level categories match (Usually catches obvious errors)
    assertion_failed = False
    if assertion:
        if fpc not in FPG2FPC[fpg]:
            assertion_failed = True
        if fpg != "Meals" and fpc != pfpc:
            assertion_failed = True
        if fpg != "Meals" and pfpc not in FPG2FPC[fpg]:
            assertion_failed = True

    if fpc == "Non-Food" or fpg == "Non-Food":
        legible_preds = {
            "Food Product Group": "Non-Food",
            "Food Product Category": "Non-Food",
            "Primary Food Product Category": "Non-Food",
        }
        for key in model.config.columns.keys():
            if key not in legible_preds:
                legible_preds[key] = None
    else:
        legible_preds = {}
        for item, score in zip(model.decoders.items(), scores):
            col, decoder = item
            prob, idx = score

            if col != "Sub-Types":
                pred = decoder[str(idx.item())]  # decoders have been serialized so keys are strings
                legible_preds[col] = pred if not assertion_failed else None
                if confidence_score:
                    legible_preds[col + "_score"] = prob.item()

        # Handle subtype predictions separately
        # Note: Make sure we are not predicting class "None"
        subtype_logits[:, int(model.none_subtype_idx)] = 0
        topk_values, topk_indices = torch.topk(subtype_logits, 2)
        mask = torch.zeros_like(subtype_logits)
        mask.scatter_(1, topk_indices, 1)
        subtype_logits = subtype_logits * mask
        subtype_preds = (sigmoid(subtype_logits) > threshold).int()
        predicted_subtype_indices = (
            torch.nonzero(subtype_preds.squeeze()) if not assertion_failed else torch.tensor([])
        )

        num_subtype_columns = sum(1 for key in model.config.columns.keys() if "Sub-Type" in key)
        predicted_subtype_tuples = []

        for idx in predicted_subtype_indices:
            for j in range(num_subtype_columns):
                subtype_col_idx = j + 1
                legible_subtype = model.decoders["Sub-Types"][str(idx.item())]
                if legible_subtype in model.counts[f"Sub-Type {subtype_col_idx}"]:
                    # Note: Sort tuples by presence in subtype column, then frequency
                    # TODO: Add a sort here for whether this is a "misc" column tag (put those last)
                    # Create a set of all misc tags and check for membership...
                    predicted_subtype_tuples.append(
                        (
                            subtype_col_idx,
                            model.counts[f"Sub-Type {subtype_col_idx}"][legible_subtype],
                            legible_subtype,
                        )
                    )
                    break
        # Note: sort by column index then by frequency
        predicted_subtype_tuples = sorted(predicted_subtype_tuples)
        # Move overflow misc columns to the end of the subtype list
        non_misc_subtypes = [tup for tup in predicted_subtype_tuples if tup[2] not in ALL_NON_SUBTYPE_TAGS]
        misc_subtypes = [tup for tup in predicted_subtype_tuples if tup[2] in ALL_NON_SUBTYPE_TAGS]
        predicted_subtype_tuples = non_misc_subtypes + misc_subtypes

        for i, (_, _, subtype) in enumerate(predicted_subtype_tuples):
            legible_preds[f"Sub-Type {i+1}"] = subtype

    if combine_name:
        return get_combined_name(legible_preds)
    return legible_preds


def save_output(df_classified: pd.DataFrame, filename: Union[str, Path], data_dir: Union[str, Path]) -> None:
    """Saves a DataFrame to an Excel file in the specified directory, replacing "None" values with NaN.

    Args:
        df_classified: The DataFrame to be saved.
        filename: The name of the file to save the output as. Can be a string or Path object.
        data_dir: The directory where the file should be saved. Can be a string or Path object.

    Returns:
        None
    """
    if not isinstance(filename, Path):
        filename = Path(filename)
    os.chdir(data_dir)  # ensures this saves in the expected directory in Colab
    output_path = filename.with_name(filename.stem + "_classified.xlsx")
    df_classified = df_classified.replace("None", np.nan)
    df_classified.to_excel(output_path, index=False)
    print(f"Classification completed! File saved to {output_path}")
    return


def inference_handler(
    model: Any,
    tokenizer: Any,
    input_path: Union[str, Path],
    input_column: str,
    output_filename: Optional[Union[str, Path]] = None,
    save_dir: Union[str, Path] = "/content",
    device: Optional[str] = None,
    sheet_name: Union[int, str] = 0,
    save: bool = True,
    threshold: float = 0.85,
    num_rows_to_classify: Optional[int] = None,
    assertion: bool = False,
) -> pd.DataFrame:
    """Handles the inference pipeline on a dataset, including reading data, performing inference, and saving results.

    Args:
        model: The model used for inference.
        tokenizer: The tokenizer used to preprocess the text input.
        input_path: Path to the input Excel file.
        input_column: The column name in the Excel file containing the text to classify.
        output_filename: Optional filename for the output file
        save_dir: Directory where the output file should be saved
        device: The device to run inference on
        sheet_name: The sheet name or index to read from the Excel file
        save: Whether to save the output file
        confidence_score: Whether to include confidence scores in the output
        threshold: Threshold for determining uncertainty in predictions
        num_rows_to_classify: Number of rows to classify. If None, all rows are classified.
        assertion: Whether to perform additional assertions during inference

    Returns:
        The DataFrame containing predictions from the model
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    try:
        df_input = pd.read_excel(input_path, sheet_name=sheet_name)
    except FileNotFoundError:
        print("FileNotFound: {e}\n. Please double check the filename: {input_path}")
        raise

    # Force columns to have capitalization consistent with expected output
    df_input.columns = [col.lower() for col in df_input.columns]
    df_input = df_input.rename(columns=lower2label)

    if num_rows_to_classify:
        df_input = df_input.head(num_rows_to_classify)

    output = (
        df_input[input_column]
        .apply(lambda text: inference(model, tokenizer, text, device, assertion=assertion))
        .apply(pd.Series)
    )
    output = output.replace("None", pd.NA)

    # Add all columns to results to match name normalization format
    results_full = pd.DataFrame()
    for col in OUTPUT_COLS:
        if col in output:
            results_full[col] = output[col]
        elif col in NON_LABEL_COLS:
            results_full[col] = df_input[col]
        else:
            results_full[col] = pd.Series([None] * len(output))

    if output_filename is None:
        output_filename = input_path

    if save:
        save_output(results_full, output_filename, save_dir)
    return results_full


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent

    # TODO: This is ugly...
    with Path.open(SCRIPT_DIR / "../../../scripts/config_train.yaml") as file:
        config = yaml.safe_load(file)

    DATA_DIR = Path(config["data"]["data_dir"]) / "test"
    test_filepath = DATA_DIR / config["data"]["test_filename"]
    model_checkpoint = Path(config["model"]["eval_checkpoint"])

    model = MultiTaskModel.from_pretrained(model_checkpoint)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    SHEET_NUMBER = 0
    CONFIDENCE_SCORE = False
    ROWS_TO_CLASSIFY = None
    RAW_RESULTS = False
    ASSERTION = False

    INPUT_COLUMN = config["data"]["text_field"]

    inference_handler(
        model,
        tokenizer,
        input_path=test_filepath,
        save_dir=DATA_DIR,
        device=device,
        sheet_name=SHEET_NUMBER,
        input_column=INPUT_COLUMN,
        num_rows_to_classify=ROWS_TO_CLASSIFY,
        confidence_score=CONFIDENCE_SCORE,
        assertion=ASSERTION,
    )

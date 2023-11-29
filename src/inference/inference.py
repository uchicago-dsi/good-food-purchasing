import torch
import pandas as pd
import os

def inference(model, tokenizer, text, device, confidence_score=False):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    inputs = inputs.to(device)
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
    softmaxed_scores = [torch.softmax(logits, dim=1) for logits in outputs.logits]
    scores = [torch.max(score, dim=1) for score in softmaxed_scores] # torch.max returns both max and argmax

    legible_preds = {}
    for item, score in zip(model.decoders, scores):
        col, decoder = item
        prob, idx = score
        try:
            legible_preds[col] = decoder[str(idx.item())] # decoders have been serialized so keys are strings
            if confidence_score:
                legible_preds[col + "_score"] = prob.item()
        except Exception as e:
            pass
            # TODO: what do we want to actually happen here?
            # Can we log or print base on where we are?
            # logging.info(f"Exception: {e}")
    return legible_preds

def highlight_uncertain_preds(df, threshold=.85):
  """Creates a styles dictionary for underconfident predictions"""
  styles_dict = {}
  for col_idx, dtype in enumerate(df.dtypes):
    # TODO: this is fragile - fix it later
    if dtype == 'object': # Skip non-float columns
      continue
    else:
      try:
        styles_dict[df.columns[col_idx - 1]] = df.iloc[:,col_idx].apply(lambda x: 'background-color: yellow' if x < threshold else '')
      except:
        print(f"Tried to find uncertainty in a a non-float column! {df.iloc[:,col_idx].head(5)}")
  return styles_dict

def inference_handler(model, tokenizer, input_path, input_column, device=None, sheet_name=0, confidence_score=False, threshold=.85, rows_to_classify=None):
  if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

  try:
    df = pd.read_excel(input_path, sheet_name=sheet_name)
  except FileNotFoundError:
    # TODO: make this error message clearer
    print("There was a FileNotFoundError thrown, please double-check the file name, or ensure that you have correctly uploaded the file to the Google Colab drive.")
    raise

  if rows_to_classify:
    df = df.head(rows_to_classify)

  output = df[input_column].apply(lambda text: inference(model, tokenizer, text, device, confidence_score=confidence_score)).apply(pd.Series)
  results = pd.concat([df[input_column], output], axis=1)

  # Add all columns to results to match name normalization format
  # Assumes that the input dataframe is in the expected name normalization format
  results_full = pd.DataFrame()
  for col in df.columns:
    if col in results:
      results_full[col] = results[col]
    elif col == "Center Product ID":
      results_full[col] = df[col]
    else:
      results_full[col] = pd.Series([None] * len(results))

    # Add score col if confidence_score is passed
    score_col = col + "_score"
    if confidence_score and score_col in results:
      results_full[score_col] = results[score_col]

  # highlight uncertain predictions
  if confidence_score:
    styles_dict = highlight_uncertain_preds(results_full, threshold)
    styles_df = pd.DataFrame(styles_dict)

    # Actually apply the styles — applying styles gives you a Styler object not a dataframe
    df_formatted = results_full.style.apply(lambda x: styles_df, axis=None)
  else:
    df_formatted = results_full

  # TODO: maybe we want option to have styles but not have scores?

  os.chdir("/content/") # make sure this saves in the expected directory
  df_formatted.to_excel(input_path + "_classifed.xlsx", index=False)
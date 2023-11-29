import torch

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
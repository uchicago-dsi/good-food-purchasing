import sys
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, TrainingArguments, Trainer, PreTrainedModel, DistilBertConfig, DistilBertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import polars as pl
from datetime import datetime

logging.basicConfig(level=logging.INFO)

### Setup

MODEL_NAME = 'distilbert-base-uncased'
TEXT_FIELD = "Product Type"
LABELS = [
    "Food Product Category",
    "Level of Processing",
    "Primary Food Product Category",
    "Food Product Group",
    "Primary Food Product Group"
]
# TODO: set up some sort of way to have a model repo when the models are actually good
# MODEL_PATH = f"/net/projects/cgfp/saved-models/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
MODEL_PATH = f"/net/projects/cgfp/results/cgfp-classifier/five-cols"

SMOKE_TEST = False
SAVE_BEST = True # TODO: Need to actually set up eval metric for this to be a good idea

if SMOKE_TEST:
    MODEL_PATH += "smoke-test"

# TODO: move models into separate file

class MultiTaskConfig(DistilBertConfig):
    def __init__(self, num_categories_per_task=None, decoders=None, columns=None, **kwargs):
        super().__init__(**kwargs)
        self.num_categories_per_task = num_categories_per_task
        self.decoders = decoders
        self.columns = columns

class MultiTaskModel(PreTrainedModel):
    config_class = MultiTaskConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.num_categories_per_task = config.num_categories_per_task
        self.decoders = config.decoders
        self.columns = config.columns

        self.classification_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.dim, config.dim),
                nn.ReLU(),
                nn.Dropout(config.seq_classif_dropout),
                nn.Linear(config.dim, num_categories)
            ) for num_categories in self.num_categories_per_task
        ])

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]

        logits = [classifier(pooled_output) for classifier in self.classification_heads]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            losses = []
            for logit, label in zip(logits, labels.squeeze().transpose(0, 1)): # trust me
                losses.append(
                    loss_fct(logit, label.view(-1))
                )
            loss = sum(losses)

        output = (logits,) + distilbert_output[1:] # TODO why activations? see https://github.com/huggingface/transformers/blob/6f316016877197014193b9463b2fd39fa8f0c8e4/src/transformers/models/distilbert/modeling_distilbert.py#L824
    
        if not return_dict:
            return (loss,) + output if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions
        )

def read_data(data_path):
    return pl.read_csv(data_path, infer_schema_length=1, null_values=['NULL']).lazy()


### Data prep

# Training

def compute_metrics(pred):
    # Extract the predictions and labels for each task

    # TODO: set this up so that it gets all of the labels and predictions in 
    # a principled and scalable way for an arbitrary number of columns
    
    # TODO: fix predictions
    # len(pred.predictions) » 2
    # len(pred.predictions[0]) » 20
    # len(pred.predictions[0][0]) » 6 (number of classes)
    # len(pred.predictions[1][0]) » 29 (number of classes)
    # Also...why is this 20 and not the batch size?

    preds_task1 = pred.predictions[0].argmax(-1)
    preds_task2 = pred.predictions[1].argmax(-1)

    # TODO: fix label_ids
    # len(pred.label_ids) » 2
    # This comes in a list of length 20 with a 2D label for each example?
    # array([[ 5],
    #    [26]])
    labels_task1 = pred.label_ids[:, 0, 0].tolist()
    labels_task2 = pred.label_ids[:, 1, 0].tolist()

    # Compute metrics for Task 1
    accuracy_task1 = accuracy_score(labels_task1, preds_task1)
    # precision_task1, recall_task1, f1_task1, _ = precision_recall_fscore_support(labels_task1, preds_task1, average='macro')

    # Compute metrics for Task 2
    accuracy_task2 = accuracy_score(labels_task2, preds_task2)
    # precision_task2, recall_task2, f1_task2, _ = precision_recall_fscore_support(labels_task2, preds_task2, average='macro')

    # TODO: hack to get something working
    composite_metric = (accuracy_task1 + accuracy_task2) / 2

    # Return a dictionary of combined metrics
    return {
        "composite_metric": composite_metric
    }

if __name__ == '__main__':

    logging.info(f"MODEL_PATH : {MODEL_PATH}")

    # Setup

    logging.info("Starting")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info(f"Training with device : {device}")

    logging.info(f"Using base model : {MODEL_NAME}")
    logging.info(f"Predicting based on input field : {TEXT_FIELD}")
    logging.info(f"Predicting categorical fields : {LABELS}")

    # Data preparation

    # TODO: Should we force everything into the name normalization format here?
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data'
    logging.info(f"Reading data from path : {data_path}")
    df = read_data(data_path)
    columns = df.columns
    df_cleaned = df.select(TEXT_FIELD, *LABELS)
    df_cleaned = df_cleaned.drop_nulls()

    encoders = {}
    for column in LABELS:
        encoder = LabelEncoder()
        encoder.fit_transform(df_cleaned.select(column).collect().to_numpy().ravel())
        encoders[column] = encoder

    # Create decoders to save to model config
    decoders = {}
    for col, encoder in encoders.items():
        decoding_dict = {index: label for index, label in enumerate(encoder.classes_)}
        decoders[col] = decoding_dict
        logging.info(f"{col}: {len(encoder.classes_)} classes")

    logging.info("Preparing dataset")
    dataset = Dataset.from_pandas(df_cleaned.collect().to_pandas())

    if SMOKE_TEST:
        dataset = dataset.select(range(1000))

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    def tokenize(batch):
        tokenized_inputs = tokenizer(batch[TEXT_FIELD], padding='max_length', truncation=True, max_length=100)
        tokenized_inputs["labels"] = [encoders[label].transform([batch[label]]) for label in LABELS]
        return tokenized_inputs
    dataset = dataset.map(tokenize)

    dataset.set_format('torch', columns=['input_ids', 'attention_mask', "labels"])
    dataset = dataset.train_test_split(test_size=0.2)
    logging.info("Dataset is prepared")

    logging.info(f"Structure of the dataset : {dataset}")
    logging.info(f"Sample record from the dataset : {dataset['train'][0]}")

    # Training

    logging.info("Instantiating model")
    distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    num_categories_per_task = [len(v.classes_) for k, v in encoders.items()]
    config = MultiTaskConfig(num_categories_per_task=num_categories_per_task, decoders=decoders, columns=columns, **distilbert_model.config.to_dict())
    model = MultiTaskModel(config)
    logging.info("Model instantiated")

    epochs = 5 if SMOKE_TEST else 50

    training_args = TrainingArguments(
        output_dir = '/net/projects/cgfp/checkpoints',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs = epochs,
        per_device_train_batch_size = 32,
        per_device_eval_batch_size = 64,
        warmup_steps = 100,
        weight_decay = 0.01,
        logging_dir = './training-logs'
    )

    if SAVE_BEST:
        training_args.load_best_model_at_end=True
        training_args.metric_for_best_model='composite_metric'
        training_args.greater_is_better=True

    adamW = AdamW(model.parameters(), lr=0.0003)
    trainer = Trainer(
        model = model,
        args = training_args,
        compute_metrics = compute_metrics, 
        train_dataset = dataset['train'],
        eval_dataset = dataset['test'],
        optimizers = (adamW, None)  # Optimizer, LR scheduler
    )

    logging.info("Training...")
    trainer.train()

    logging.info("Saving the model")
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

    logging.info("Complete!")
    # TODO: add example output

    def inference(model, text, device, confidence_score=False):
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

        inputs = inputs.to(device)
        model = model.to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
        logging.info(f"outputs: {outputs}")
        scores = [torch.max(logits, dim=1) for logits in outputs['logits']] # torch.max returns both max and argmax

        legible_preds = {}
        for item, score in zip(model.decoders.items(), scores):
            col, decoder = item
            prob, idx = score
            try:
                # TODO: should prob deserialize the ints for the notebook...
                legible_preds[col] = decoder[idx.item()]
                if confidence_score:
                    legible_preds[col + "_score"] = prob.item()
            except Exception as e:
                # TODO: what do we want to actually happen here?
                logging.info(f"Exception: {e}")

        return legible_preds

    prompt = "frozen peas and carrots"
    legible_preds = inference(model, prompt, device)
    logging.info(f"Example output for 'frozen peas and carrots': {legible_preds}")

import sys
import logging
import json
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import polars as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, TrainingArguments, Trainer, PreTrainedModel, DistilBertConfig, DistilBertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset

from inference.inference import inference
from training.models import MultiTaskConfig, MultiTaskModel

logging.basicConfig(level=logging.INFO)

### Setup

MODEL_NAME = 'distilbert-base-uncased'
TEXT_FIELD = "Product Type"
# TODO: change this default somewhere
LABELS = [
    "Food Product Category",
    "Level of Processing",
    "Primary Food Product Category",
    "Food Product Group",
    "Primary Food Product Group"
]
# TODO: add args to MODEL_PATH and logging path
MODEL_PATH = f"/net/projects/cgfp/model-files/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"

SMOKE_TEST = False
SAVE_BEST = True

if SMOKE_TEST:
    MODEL_PATH += "-smoke-test"

def read_data(data_path):
    return pl.read_csv(data_path, infer_schema_length=1, null_values=['NULL']).lazy()

### Data prep

# Training

def compute_metrics(pred):
    '''
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
    '''

    num_tasks = len(pred.predictions)
    preds = [pred.predictions[i].argmax(-1) for i in range(num_tasks)]
    labels = [pred.label_ids[:, i, 0].tolist() for i in range(num_tasks)]

    accuracies = {}
    for i, task in enumerate(zip(preds, labels)):
        pred, lbl = task
        accuracies[i] = accuracy_score(lbl, pred)

    mean_accuracy = sum(accuracies.values())/num_tasks

    return {
        "mean_accuracy": mean_accuracy,
        "accuracies": accuracies
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

    # TODO: Set this up so data file comes out of data pipeline
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
    # Huggingface is picky...so a list of tuples seems like the best bet
    # for saving to the config.json in a way that doesn't break when loaded
    decoders = []
    for col, encoder in encoders.items():
        decoding_dict = {f"{index}": label for index, label in enumerate(encoder.classes_)}
        decoders.append((col, decoding_dict))
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

    # TODO: set this up so that classification can be passed via args
    classification = "mlp"
    distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    num_categories_per_task = [len(v.classes_) for k, v in encoders.items()]
    config = MultiTaskConfig(num_categories_per_task=num_categories_per_task, decoders=decoders, columns=columns, classification=classification, **distilbert_model.config.to_dict())
    model = MultiTaskModel(config)
    logging.info("Model instantiated")

    epochs = 5 if SMOKE_TEST else 15

    # TODO: add an arg for freezing layers
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classification heads
    for param in model.classification_heads.parameters():
        param.requires_grad = True

    # TODO: set this up to come from args
    lr = .001

    # TODO: Training logs argument doesn't seem to work. Logs are in the normal logging folder?
    # Add info to logging file name
    training_args = TrainingArguments(
        output_dir = '/net/projects/cgfp/checkpoints',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs = epochs,
        per_device_train_batch_size = 32,
        per_device_eval_batch_size = 64,
        lr_scheduler_type='linear',
        learning_rate=lr,
        warmup_steps = 100,
        weight_decay = 0.01,
        logging_dir = './training-logs'
        # logging_dir = '/home/tnief/good-food-purchasing/training-logs'
    )

    if SAVE_BEST:
        training_args.load_best_model_at_end=True
        training_args.metric_for_best_model='mean_accuracy'
        training_args.greater_is_better=True

    # TODO: depending on how we're actually training an LR scheduler is probably useful
    # Samet hing with learning rate
    adamW = AdamW(model.parameters(), lr=lr)
    # TODO: pass this from args...maybe include an option for None
    # Should prob actually use: torch.optim.lr_scheduler.ReduceLROnPlateau
    # scheduler = ReduceLROnPlateau(adamW) doesn't work great with trainer

    # idea from ChatGPT:
    # class ReduceOnPlateauCallback(TrainerCallback):
    # def __init__(self, optimizer, *args, **kwargs):
    #     self.scheduler = ReduceLROnPlateau(optimizer, *args, **kwargs)

    # def on_evaluate(self, args, state, control, **kwargs):
    #     metrics = kwargs.get('metrics', {})
    #     validation_loss = metrics.get('eval_loss', None)
    #     if validation_loss is not None:
    #         self.scheduler.step(validation_loss)
    #
    # add this to trainer args: 
    # callbacks=[ReduceOnPlateauCallback(optimizer, mode='min', factor=0.1, patience=10)]

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

    logging.info("Training complete")
    logging.info(f"Validation Results: {trainer.evaluate()}")

    logging.info("Saving the model")
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

    logging.info("Complete!")

    prompt = "frozen peas and carrots"
    legible_preds = inference(model, tokenizer, prompt, device)
    logging.info(f"Example output for 'frozen peas and carrots': {legible_preds}")

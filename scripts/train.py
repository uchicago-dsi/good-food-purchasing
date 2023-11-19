import sys
import logging

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, TrainingArguments, Trainer, PreTrainedModel
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
    "Level of Processing",
    "Primary Food Product Category",
]
MODEL_PATH = f"/net/projects/cgfp/saved-models/model-name/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
SMOKE_TEST = True

class MultiTaskModel(PreTrainedModel):
    def __init__(self, model, num_categories_per_task, *args, **kwargs):
        config = model.config
        super().__init__(config)
        self.model = model
        self.distilbert = model.distilbert
        self.num_categories_per_task = num_categories_per_task
        self.classification_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.dim, config.dim),
                nn.ReLU(),
                nn.Dropout(config.seq_classif_dropout),
                nn.Linear(config.dim, num_categories)
            ) for num_categories in num_categories_per_task
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

        # TODO: do we actually need to implement a return_dict setup?

        return ((loss,) + output) if loss is not None else output

def read_data(data_path):
    return pl.read_csv(data_path, infer_schema_length=1, null_values=['NULL']).lazy()


### Data prep

# Training

# ????
def compute_metrics(pred):
    # Extract the predictions and labels for each task
    preds_task1 = pred.predictions[0].argmax(-1)
    preds_task2 = pred.predictions[1].argmax(-1)

    labels_task1 = pred.label_ids[0]
    labels_task2 = pred.label_ids[1]

    # Compute metrics for Task 1
    accuracy_task1 = accuracy_score(labels_task1, preds_task1)
    precision_task1, recall_task1, f1_task1, _ = precision_recall_fscore_support(labels_task1, preds_task1, average='macro')

    # Compute metrics for Task 2
    accuracy_task2 = accuracy_score(labels_task2, preds_task2)
    precision_task2, recall_task2, f1_task2, _ = precision_recall_fscore_support(labels_task2, preds_task2, average='macro')

    # TODO: hack to get something working
    composite_metric = (f1_task1 + f1_task2) / 2

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

    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data'
    logging.info(f"Reading data from path : {data_path}")
    df = read_data(data_path)
    df_cleaned = df.select(TEXT_FIELD, *LABELS)
    df_cleaned = df_cleaned.drop_nulls()

    encoders = {}
    num_labels = 0
    for column in LABELS:
        encoder = LabelEncoder()
        encoder.fit_transform(df_cleaned.select(column).collect().to_numpy().ravel())
        encoders[column] = encoder
        num_labels += len(encoder.classes_)
    logging.info(f"Number of labels to predict : {num_labels}")

    logging.info("Preparing dataset")
    dataset = Dataset.from_pandas(df_cleaned.collect().to_pandas())

    if SMOKE_TEST:
        dataset = dataset.select(range(10))

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
    model = MultiTaskModel(distilbert_model, [len(v.classes_) for k, v in encoders.items()])
    # config = DistilBertConfig()  # Use the appropriate config for your base model
    # model = MultiTaskModel(config=config, num_categories_per_task=[10, 5])
    logging.info("Model instantiated")

    training_args = TrainingArguments(
        output_dir = '/net/projects/cgfp/checkpoints',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='composite_metric',
        greater_is_better=True,
        num_train_epochs = 2,
        per_device_train_batch_size = 32,
        per_device_eval_batch_size = 64,
        warmup_steps = 100,
        weight_decay = 0.01,
        logging_dir = './training-logs'
    )

    adamW = AdamW(model.parameters(), lr=0.0003)
    trainer = Trainer(
        model = model,
        args = training_args,
        compute_metrics = compute_metrics,
        train_dataset = dataset['train'],
        eval_dataset = dataset['test'],
        optimizers = (adamW, None)  # Optimizer, LR scheduler
    )

    trainer.train()

    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

    logging.info(data_path)

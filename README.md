# Center for Good Food Purchasing Name Normalization Tool

The [Center for Good Food Purchasing](https://goodfoodpurchasing.org/) scores foods purchased by public institutions on five factors: local economies, health, valued workforce, animal welfare, and environmental sustainability.

To do this scoring, they transform the food names from purchase orders into a "normalized name" that is based on a hierarchical tagging structure. These normalized names are then scored on CGFP's five factors.

For example, ```CRANBERRIES, DRIED, INDIVIDUALLY PACKAGED (1.16 oz./pkg.)``` becomes ```cranberry, dried, ss``` and ```YOGURT, ASSORTED GREEK LIGHT SS CUP REF STRAWBERRY BLUEBERRY``` becomes ```yogurt, greek, variety, ss```.

The name normalization process is time-consuming and error-prone, so we have trained a language model to perform a first-pass of name normalization.

This repo contains: 
- A data cleaning pipeline to clean CGFP's (Center for Good Food Purchasing) historical labeled data for use in training a text classifier to perform their name normalization task. **This is designed to be run locally.**
- A training pipeline to train a text classifier to perform the name normalization task. **This is to be run on UChicago's DSI Cluster.**

## Pipeline

CGFP has provided exports from their scoring platform that include tens thousands of examples with human-labeled normalized names.

However, there are lots of inconsistencies in the data and CGFP's name normalization requirements have changed over time.

Also, our text classifier is a multi-task classifer, that performs classification across multiple columns, so we need to split the normalized names into their appropriate columns.

The data pipeline takes in the normalized name (eg ```taquito, egg, turkey sausage, cheese, potato, frozen```) and splits it into multiple columns following the structure in CGFP's [name normalization helper](https://docs.google.com/spreadsheets/d/18Gvb_PlcRyOWidXCmgaIEgpsnvqVJYyC/edit?usp=sharing&ouid=114633865943391212776&rtpof=true&sd=true).

### Quickstart
- Build the Docker container to run the pipeline (see the [Docker](#docker) section)
- Download the raw data you wish to clean (see the [Raw Data](#raw-data) section)
- Update ```scripts/config_pipeline.yaml``` with the filename and location of the data you wish to clean. Note that the file is expected to be in `/data/raw` _inside_ the container.
- Run ```scripts/pipeline.py```

### Understanding the Data Pipeline

While the high-level idea of the data pipeline is relatively intuitive, the implementation is messy and full of one-offs and edge cases so it can be a bit hard to follow what's going on.

The basic intuition goes something like this:
- We split the normalized name on commas so we have a list of tags
- We process the tags one by one, allocating them to the appropriate name normalization column
  - The first tag in a normalized name is always "Basic Type"
  - Different columns have different allowed tags based on the product's Food Product Group and Food Product Category: we check if a tag is allowed in any of the other columns, and, if so, allocate it to that column
  - If a tag is not allocated to any of the other columns, allocate it to a "Sub-Type" column
  - Throughout the process, we check for edge cases and directly apply any rules associated with these edge cases

The allowed tags for each column are saved in ```misc_tags.py```

Much of the rest of the pipeline code is handling edge cases. Most of these rules are saved as dictionaries in ```src/cgfp/constants/tokens```

### Docker

The data pipeline runs in Docker. 

If you are using VS Code, there is a ```.devcontainer``` folder with the Docker configuration to run as a dev container.

Otherwise, build the Docker image using the ```Dockerfile``` in the root of the repo.

### Raw Data

We've been using the pipeline to clean these two data sets:
- <a href="https://docs.google.com/spreadsheets/d/1c5v7nBhqQpjOb7HE7pqDUx_xMc8r1imc/edit?usp=sharing&ouid=114633865943391212776&rtpof=true&sd=true" target="_blank">CONFIDENTIAL_CGFP bulk data_073123</a>
- <a href="https://docs.google.com/spreadsheets/d/1PziC9jR8yHQex9RB49JoH5s4nXd1fLoK/edit?usp=sharing&ouid=114633865943391212776&rtpof=true&sd=true" target="_blank">New_Raw</a>

Make sure to download them as `xlsx` files from the google drive site. The files are expected to be the `/data/raw` directory.

### Clean Data

Once the data cleaning pipeline is run, files should appear in the directory `data/clean/pipeline-{date}-{time}`. These files are the input for the training / text classifier. The following files are created:

```
value_counts.xlsx
normalized_name_diff.csv
scoring.csv
misc.csv
clean_CONFIDENTIAL_CGFP_bulk_data_073123.csv
```

The final file of this list is the one that is required for the classification task below.

## Text Classifier

We use Huggingface to train a multi-task text classifier on the name normalization task.

We take in an example's Product Type as input (eg ```CRANBERRIES, DRIED, INDIVIDUALLY PACKAGED (1.16 oz./pkg.)```) and we output a classification for each column in CGFP's [name normalization tool](https://docs.google.com/spreadsheets/d/18Gvb_PlcRyOWidXCmgaIEgpsnvqVJYyC/edit?usp=sharing&ouid=114633865943391212776&rtpof=true&sd=true).

Note that, other than "Food Product Group", "Food Product Category", "Primary Food Product Category", and "Basic Type", all of the other columns can be (and usually are) empty.

We have infrastructure to train both RoBERTa and DistilBERT models.

**Before running the below make sure that you set up an environment variable with your [wandb.ai api key](https://wandb.ai/home). The variable should be set as `WANDB_API_KEY` and should be available in the environment that is calling the `make` commands below.**

**There is configuration variable in the `config_train.yaml` called `smoke_test`. Setting this to `true` will run through a small subset of the data in order to test any code. You need to set it to `False` in order to run the actual models.***

### Training the Classifier

To get good results for all columns, we need to do a multi-stage fine-tuning process.

- First, clean a dataset using the [data pipeline](#pipeline). Upload this dataset to wherever you'll be training your model. 
- Upload a validation set and a test set
  - [Validation set](https://docs.google.com/spreadsheets/d/1pyEBLXbNEDH4D0k7y94jR3pG2z29qgxD3Iqmyfjiblc/edit?usp=sharing). Note that this should be downloaded as a CSV file.
  - [Test set](https://docs.google.com/spreadsheets/d/1em0DvmnjTu3h7NfTjf1DdJtl8axuLeL3/edit?usp=sharing&ouid=114633865943391212776&rtpof=true&sd=true) (Note: this is not a "true" test set, but is used at the end of training to run inference with the trained model) This should be downloaded as an Excel File.
- Update ```scripts/config_train.yaml``` with the location of your training, validation, and testing datasets and the location where you'd like to save your models.
  - All files should be in the same directory (`data_dir` in the `config_train.yaml`).
  - You can also choose training options in this yaml file. Most of the defaults should work well.
- Build the conda environment in ```environment.yml```. Note that this creates an environment with the name `cgfp`, as per the [file](environment.yml).
```bash 
conda env create -f environment.yml
```
- Create a `.env` file in the root directory. You will need to create a `.env` file which contains the following information:
    ```
    SBATCH_MAIL=your_cnet_id@gmail.com
    CONDA_ENV_PATH=full path to conda location
    CGFP_DIR=/net/projects/cgfp/
    DSI_PARTITION=general
    ENV_NAME=cgfp
    ```
  - The `CONDA_ENV_PATH` is specific to how you installed conda. You should look for a directory which contains the `cfgp` environment that was created by the previous `environment.yml`. If you aren't sure where to look you can try typing `echo $CONDA_PREFIX` into the terminal to find the root. On my installation:
    ```
    echo $CONDA_PREFIX
      /home/nickross/miniconda3
    (base) nickross@fe01:~/miniconda3/envs/cgfp$ ls /home/nickross/miniconda3/envs/cgfp/
    ```
    and I use `/home/nickross/miniconda3/envs/cgfp/` as my `CONDA_ENV_PATH`

- Then, train the model using the following command.
  ```bash
  make train
  ```
  - If you are _not_ using the UChicago DSI cluster, activate the ```cgfp``` conda environment and run ```scripts/train.py```

#### Multi-Stage Fine-Tuning

To get good performance across tasks, we run a multi-stage fine-tuning process where we freeze and unfreeze the base model while also attaching different classification heads to the computation graph. We can configure all of this in the ```config_train.yaml```.

We use the same command as above (`make train`) on the DSI cluster to run the training. Updating the `config_train.yaml` file changes how the training is run; it is basically a conf file which controls all aspects of the training.

We start by training the entire model on "Basic Type" while detaching all other classification heads from the computation graph (so they do not impact the representations from the base model). To do this, we set the following settings in ```config_train.yaml``` (leaving all other rows unchanged.):

You want to use either the roberta or distilbert models, make sure to adjust the learning rate parameter in the `config_train.yaml` accordingly.

```
model:
  freeze_base: false
  attached_heads:
    - "Basic Type"
training:
  metric_for_best_model: "basic_type_accuracy"
training:
  lr: 2e-5 # .001 for distilbert, 2e-5 for roberta
```

Once this file is updated re-run `make train`.

Next, we load the model trained on "Basic Type" only and train the full model on "Sub-Types".

```
model:
  starting_checkpoint: path/to/basic/type/trained/model
  freeze_base: false
  attached_heads:
    - "Sub-Types"
training:
  metric_for_best_model: "mean_f1_score"
training:
  lr: 2e-5 # .001 for distilbert, 2e-5 for roberta  
```

The results after these two steps are usually quite good.

When running the above you want to pick one of roberta or distilbert and then run the first which will generate the starting checkpoint for use in the second. **Make sure that the models in the starting\_checkpoint align**

#### Training Just the Classification Heads

If we have further cleaned the data and would like to just retrain the classification heads (without retraining the base model), we can train the model with the following settings:

```
model:
  starting_checkpoint: path/to/fine/tuned/model
  freeze_base: true
  reset_classification_heads: true
  attached_heads: null  # Doesn't matter since base is frozen
training:
  metric_for_best_model: "mean_f1_score"
```

### Inference

We will typically be running inference on a spreadsheet of food labels. The output is set up to match CGFP's name normalization helper.

To run inference:
- Load the model and tokenizer:
```
model = MultiTaskModel.from_pretrained("uchicago-dsi/cgfp-roberta")
tokenizer = AutoTokenizer.from_pretrained("uchicago-dsi/cgfp-roberta")
```
- Use the ```inference_handler``` function:
```
inference_handler(model, tokenizer, input_path=INPUT_PATH, save_dir=DATA_DIR, device=device, sheet_name=SHEET_NUMBER, input_column="Product Type", assertion=True)
```

An example Colab notebook to run inference is [available here](https://colab.research.google.com/drive/1c8_WGWxeVCY60-luqWPiRPQr6omdbfzK?usp=sharing).

### Additional Training Notes

#### Multi-Task & Multi-Label Training

The CGFP model is set up to predict multiple different columns from a single input string. For most of these columns, the model will make a single prediction (often including leaving the column empty).

However, for Sub-Types, we are doing [multi-label prediction](https://machinelearningmastery.com/multi-label-classification-with-deep-learning/). This is because a single item can have multiple sub-types, and, while there is some ordering to the sub-types, it is inconsistent. So, we have a single Sub-Types multi-label classification head that can predict multiple sub-types. Any predicted sub-types are allocated to sub-type columns during inference.

Note that Huggingface is set up reasonably well to handle multi-task classification and multi-label classification separately, but combining the two required some customization.

We tried to document the logic for doing this, but there are several non-standard things happening in the ```forward``` method of the model (and the loss functions) in order to do multi-task learning with one head being a multi-label head.

#### Inference-Time Assertions

The model occasionally makes absurd predictions. These are usually from inputs that are outside of anything it has seen during training. We can usually catch these by noticing when "Food Product Group", "Food Product Category" & "Primary Food Product Category" do not make sense together.

If ```assertion=True``` is passed to ```inference_handler```, a blank row will be outupt for any prediction where any of the outputs for  "Food Product Group", "Food Product Category", and "Primary Food Product Category" are mismatched. Pass ```assertion=False``` to disable this behavior.

### Updating the Production Model

We host the production versions of the models on huggingface at ```uchicago-dsi/cgfp-distilbert``` and ```uchicago-dsi/cgfp-roberta```.

There are commands in the ```Makefile``` to update the models hosted on Huggingface. Make sure the performance on these is good and stable before updating since CGFP is actively using these models!

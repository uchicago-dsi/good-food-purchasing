# Center for Good Food Purchasing Name Normalization Tool

The [Center for Good Food Purchasing](https://goodfoodpurchasing.org/) scores foods purchased by public institutions on five factors: local economies, health, valued workforce, animal welfare, and environmental sustainability.

To do this scoring, they transform the food names from purchase orders into a "normalized name" that is based on a hierarchical tagging structure. These normalized names are then scored on CGFP's five factors.

For example, ```CRANBERRIES, DRIED, INDIVIDUALLY PACKAGED (1.16 oz./pkg.)``` becomes ```cranberry, dried, ss``` and ```YOGURT, ASSORTED GREEK LIGHT SS CUP REF STRAWBERRY BLUEBERRY``` becomes ```yogurt, greek, variety, ss```.

The name normalization process is time-consuming and error-prone, so we have trained a language model to perform a first-pass of name normalization.

This repo contains: 
- A data cleaning pipeline to clean CGFP's (Center for Good Food Purchasing) historical labeled data for use in training a text classifier to perform their name normalization task
- A training pipeline to train a text classifier to perform the name normalization task

## Pipeline

CGFP has provided exports from their scoring platform that include tens thousands of examples with human-labeled normalized names.

However, there are lots of inconsistencies in the data and CGFP's name normalization requirements have changed over time.

Also, our text classifier is a multi-task classifer, that performs classification across multiple columns, so we need to split the normalized names into their appropriate columns.

The data pipeline takes in the normalized name (eg ```taquito, egg, turkey sausage, cheese, potato, frozen```) and splits it into multiple columns following the structure in CGFP's [name normalization helper](https://docs.google.com/spreadsheets/d/18Gvb_PlcRyOWidXCmgaIEgpsnvqVJYyC/edit?usp=sharing&ouid=114633865943391212776&rtpof=true&sd=true).

### Quickstart

TODO

### Understanding the Data Pipeline

While the high-level idea of the data pipeline is relatively intuitive, the implementation is messy and full of one-offs and edge cases so it can be a bit hard to follow what's going on.

The basic intuition goes something like this:
- We split the normalized name on commas so we have a list of tags
- We process the tags one by one, allocating them to the appropriate name normalization column
  - The first tag in a normalized name is always "Basic Type"
  - Different columns have different allowed tags based on the product's Food Product Group and Food Product Category: we check if a tag is allowed in any of the other columns, and, if so, allocate it to that column
  - If a tag is not allocated to any of the other columns, allocate it to a "Sub-Type" column

The allowed tags for each column are saved in ```misc_tags.py```

Much of the rest of the pipeline code is handling edge cases. Many of the edge cases and pipeline rules are saved in ```src/cgfp/constants/tokens```

### Docker

The data pipeline runs in Docker. 

If you are using VS Code, there is a ```.devcontainer``` folder with the Docker configuration to run as a dev container.

Otherwise, build the Docker image using the ```Dockerfile``` in the root of the repo.

### Raw Data

We've been using the pipeline to clean these two data sets:
- [CONFIDENTIAL_CGFP bulk data_073123](https://docs.google.com/spreadsheets/d/1c5v7nBhqQpjOb7HE7pqDUx_xMc8r1imc/edit?usp=sharing&ouid=114633865943391212776&rtpof=true&sd=true)
- [New_Raw_Data_030724](https://docs.google.com/spreadsheets/d/1PziC9jR8yHQex9RB49JoH5s4nXd1fLoK/edit?usp=sharing&ouid=114633865943391212776&rtpof=true&sd=true)



# Quickstart

Run the command `make train` to beging training on a GPU node with 64gb RAM.
By default this will train on the dataset `bulk_data.csv` in the `data` folder.

To see the logs for the current or most recent run, run the command `make last-errs` (logging statements are printed to the error log, while print statements appear to be printed to the output log, so the logging I added goes to errors).

Running `make train` performs some logging setup and runs an `sbatch` command that creates a job for `./scripts/train.slurm`.
The script `./scripts/train.slurm` simply runs `./scripts/train.py`.
Checkpoints will be saved in `./results`.

For one off commands, preface your command with `conda run -p ./tmp/conda/cgfp ...`. For instance:

```conda run -p ./tmp/conda/cgfp conda list```

to list all packages in the conda environment.
Alternatively, you can start the environment.

# Environment
Libraries are installed inside a conda environment saved at `./tmp/conda/cgfp`.
Make commands (largely inspired by [this Makefile](https://github.com/conda/conda-build/blob/main/Makefile) will ensure that (a) the environment is started and the proper command is executed inside of it, (b) all changes to `environment.yml` are pushed to the conda environment, (c) dev requirements recorded in `requirements.dev.txt` are applied to the environment, and (d) local packages saved inside of `src` are installed inside the conda environment.
The PHONY target `env` ensures everything is up to date, and commands I want to execute in the conda environment (eg. `test` and `train`) simply take `env` as a prerequesite.

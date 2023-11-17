# good-food-purchasing

Multi-task classification model for food products based on their name

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

#!/usr/bin/bash
#SBATCH --job-name=cgfp-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=END
#SBATCH --constraint="a100|h100"

# TODO: should conda environment handling be done here?

python scripts/train.py
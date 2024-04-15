#!/usr/bin/bash
#SBATCH --job-name=cgfp-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=END
#SBATCH --constraint="a100|h100"

# TODO: can probably force the correct environment here?

python scripts/train.py /net/projects/cgfp/data/clean/clean_CONFIDENTIAL_CGFP_bulk_data_073123.csv
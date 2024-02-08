#!/usr/bin/bash
#SBATCH --job-name=cgfp-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=END

# python scripts/train.py /net/projects/cgfp/data/bulk_data.csv
python scripts/train.py /net/projects/cgfp/data/clean_tags.csv
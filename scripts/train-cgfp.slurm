#!/usr/bin/bash
#SBATCH --job-name=cgfp-train
#SBATCH	--gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH	--mem=64GB
#SBATCH --mail-type=END
#SBATCH --constraint="a100|h100"
#SBATCH --time=10:00:00

python scripts/train.py

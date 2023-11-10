#!/usr/bin/bash
#SBATCH --mail-type=END


conda activate cgfp
echo "Hello world"
python scripts/train.py


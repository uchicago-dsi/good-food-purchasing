from pathlib import Path

import pandas as pd
import yaml

from cgfp.constants.tokens.misc_tags import FPC2FPG

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "../data"
RAW_DIR = DATA_DIR / "raw"

with Path.open(SCRIPT_DIR / "config_pipeline.yaml") as file:
    config = yaml.safe_load(file)


df = pd.read_csv()

df = pd.read_csv("path/to/your/data.csv")  # Replace with the path to your data
df["Food Product Group"] = df.map(FPC2FPG)


breakpoint()

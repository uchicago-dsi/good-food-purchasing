"""Quick, hacky script to add Food Product Group column to datasets that are missing it"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from cgfp.constants.tokens.misc_tags import FPC2FPG

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "../data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"

with Path.open(SCRIPT_DIR / "config_pipeline.yaml") as file:
    config = yaml.safe_load(file)

FILENAME = config["input_data"]["input_file"]

RUN_FOLDER = f"fpg-{FILENAME}-{datetime.now().strftime('%Y-%m-%d %H-%M')}/".replace(" ", "_")
RUN_PATH = CLEAN_DIR / RUN_FOLDER
RUN_PATH.mkdir(parents=True, exist_ok=True)

df_cgfp = pd.read_csv(RAW_DIR / FILENAME)

df_cgfp["Food Product Group"] = df_cgfp["Food Product Category"].map(FPC2FPG)
df_cgfp["Commodity"] = None if "Commodity" not in df_cgfp.columns else df_cgfp["Commodity"]

df_cgfp.to_csv(RUN_PATH / FILENAME, index=False)

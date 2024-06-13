from datetime import datetime
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
DATA_FOLDER = project_root / "data"
RAW_FOLDER = DATA_FOLDER / "raw"
CLEAN_FOLDER = DATA_FOLDER / "clean"
RUN_FOLDER = f"pipeline-{datetime.now().strftime('%Y-%m-%d %H-%M')}/".replace(" ", "_")
RUN_FOLDER = CLEAN_FOLDER / RUN_FOLDER

DATA_FOLDER.mkdir(parents=True, exist_ok=True)
RAW_FOLDER.mkdir(parents=True, exist_ok=True)
CLEAN_FOLDER.mkdir(parents=True, exist_ok=True)
RUN_FOLDER.mkdir(parents=True, exist_ok=True)

CLEAN_FILE_PREFIX = "clean_"

GROUP_COLUMNS = [
    "Product Type",
    "Product Name",
    "Food Product Group",
    "Food Product Category",
    "Primary Food Product Category",
]

# TODO: Sub type 3?
SUBTYPE_COLUMNS = ["Sub-Type 1", "Sub-Type 2"]
NON_SUBTYPE_COLUMNS = [
    "Flavor/Cut",
    "Shape",
    "Skin",
    "Seed/Bone",
    "Processing",
    "Cooked/Cleaned",
    "WG/WGR",
    "Dietary Concern",
    "Additives",
    "Dietary Accommodation",
    "Frozen",
    "Packaging",
    "Commodity",
]
NORMALIZED_COLUMNS = ["Basic Type"] + SUBTYPE_COLUMNS + NON_SUBTYPE_COLUMNS

COLUMNS_ORDER = [
    "Product Type",
    "Food Product Group",
    "Food Product Category",
    "Primary Food Product Category",
    "Product Name",
] + NORMALIZED_COLUMNS

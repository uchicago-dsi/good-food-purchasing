# TODO: All this stuff should be moved to a config folder

from datetime import datetime
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "../..")
DATA_FOLDER = os.path.join(project_root, "data/")
RAW_FOLDER = os.path.join(DATA_FOLDER, "raw/")
CLEAN_FOLDER = os.path.join(DATA_FOLDER, "clean/")
RUN_FOLDER = f"pipeline-{datetime.now().strftime('%Y-%m-%d %H-%M')}/".replace(" ", "_")

GROUP_COLUMNS = [
    "Product Type",
    "Product Name",
    "Food Product Group",
    "Food Product Category",
    "Primary Food Product Category",
]

# TODO: Sub type 3?
NORMALIZED_COLUMNS = [
    "Sub-Type 1",
    "Sub-Type 2",
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

COLUMNS_ORDER = (
    [
        "Product Type",
        "Food Product Group",
        "Food Product Category",
        "Primary Food Product Category",
        "Product Name",
    ]
    + ["Basic Type"]
    + NORMALIZED_COLUMNS
)

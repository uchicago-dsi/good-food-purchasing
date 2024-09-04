"""Constants and configurations for training CGFP models."""

# Note: Be careful with capitalization here
FPG_FPC_COLS = ["Food Product Group", "Food Product Category", "Primary Food Product Category"]
SUB_TYPE_COLS = ["Sub-Type 1", "Sub-Type 2", "Sub-Type 3"]
MISC_COLS = [
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

# TODO: Check that this doesn't break training...
LABELS = FPG_FPC_COLS + ["Basic Type"] + SUB_TYPE_COLS + MISC_COLS

COMPLETE_LABELS = ["Product Type", "Center Product ID"] + LABELS

lower2label = {label.lower(): label for label in COMPLETE_LABELS}

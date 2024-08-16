"""Constants and configurations for training CGFP models."""

# Note: Be careful with capitalization here
FPG_FPC_COLS = ["Food Product Group", "Food Product Category", "Primary Food Product Category"]
SUB_TYPE_COLS = ["Sub-Type 1", "Sub-Type 2"]
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

LABELS = FPG_FPC_COLS + ["Basic Type"] + MISC_COLS + SUB_TYPE_COLS

# TODO: Maybe don't need these if I get them from the model.config
# These indeces are used in model configuration
# FPG_IDX = LABELS.index("Food Product Group")
# BASIC_TYPE_IDX = LABELS.index("Basic Type")
# SUB_TYPE_IDX = LABELS.index("Sub-Type 1")

COMPLETE_LABELS = ["Product Type", "Center Product ID"] + LABELS

lower2label = {label.lower(): label for label in COMPLETE_LABELS}

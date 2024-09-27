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

NON_LABEL_COLS = ["Product Identifier", "Product Type"]
LABELS = FPG_FPC_COLS + ["Basic Type"] + MISC_COLS + SUB_TYPE_COLS
OUTPUT_COLS = NON_LABEL_COLS + FPG_FPC_COLS + ["Basic Type"] + SUB_TYPE_COLS + MISC_COLS

lower2label = {label.lower(): label for label in OUTPUT_COLS}
# Note: Handle this column separately since it exists in some datasets — will be renamed in inference handler
lower2label["center product id"] = "Center Product ID"

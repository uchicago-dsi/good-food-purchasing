# TODO: All of these labels and columns should be organized better...

# Note: Be careful with capitalization here
LABELS = [
    "Food Product Group",
    "Food Product Category",
    "Primary Food Product Category",
    "Basic Type",
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

COMPLETE_LABELS = ["Product Type", "Center Product ID"] + LABELS

lower2label = {label.lower(): label for label in COMPLETE_LABELS}
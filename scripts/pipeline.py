import pandas as pd

from config import TAGS, ADDED_TAGS, TOKEN_MAP_DICT

FILEPATH = "good-food-purchasing/data/bulk_data.csv"

NORMALIZED_COLUMNS = [
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


def clean_df(df):
    # Cleaning:
    # - Remove null and short (usually a mistake) Product Types
    # - Remove null and short (usually a mistake) Product Names
    # - Remove non-food items
    df = df[
        (df["Product Type"].str.len() >= 3)
        & (df["Product Name"].str.len() >= 3)
        & (df["Food Product Group"] != "Non-Food")
    ].reset_index(drop=True)
    return df


def clean_name(name_list, food_product_category, tags_dict):
    normalized_name = {}
    misc_col = {"Misc": []}  # make a list so we can append unmatched tokens
    for i, token in enumerate(name_list):
        token = token.strip()
        if token in token_map_dict:
            token = token_map_dict[token]
        # First token is always Basic Type
        if i == 0:
            normalized_name["Basic Type"] = token
            continue
        # Check if token is in tags — if so, enter the tagging loop
        if token in tags_dict[food_product_category]["All"]:
            matched = False
            for col in NORMALIZED_COLUMNS:
                # Find the category that the token is in and add to normalized_name
                if col in tags_dict[food_product_category]:
                    if token in tags_dict[food_product_category][col]:
                        normalized_name[col] = token
                        matched = True
                        break
            if matched:
                continue
        # First token after basic type is sub-type 1 if it's not a later token
        if "Sub-Type 1" not in normalized_name:
            normalized_name["Sub-Type 1"] = token
            continue
        elif "Sub-Type 2" not in normalized_name:
            normalized_name["Sub-Type 2"] = token
            continue
        # Aggregate unmatched tokens to add to tag dictionary
        misc_col["Misc"].append(token)
    normalized_name.update(misc_col)
    # Make sure all columns are represented
    for col in NORMALIZED_COLUMNS:
        if col not in normalized_name:
            normalized_name[col] = None
    return normalized_name


combined_tags = {}
for group, group_dict in TAGS.items():
    combined_tags[group] = group_dict
    all_set = set()
    for col, tag_set in group_dict.items():
        tags_to_add = ADDED_TAGS[group][col]
        combined_set = tag_set | tags_to_add
        combined_tags[group][col] = combined_set
        all_set |= combined_set
    combined_tags[group]["All"] = all_set

df = pd.read_csv(FILEPATH)
df["Misc"] = None

df = clean_df(df)

name_dict = df.apply(
    lambda row: clean_name(
        row["Product Name"].split(","), row["Food Product Group"], combined_tags
    ),
    axis=1,
)

new_columns = pd.DataFrame(name_dict.tolist()).reset_index(drop=True)

df_split = pd.concat(
    [
        df[
            [
                "Product Type",
                "Product Name",
                "Food Product Group",
                "Food Product Category",
                "Primary Food Product Category",
            ]
        ],
        new_columns,
    ],
    axis=1,
)

misc = df_split[df_split["Misc"].apply(lambda x: x != [])][
    [
        "Product Type",
        "Food Product Group",
        "Basic Type",
        "Sub-Type 1",
        "Sub-Type 2",
        "Misc",
    ]
]

misc.to_csv("missing-tags.csv", index=False)

COLUMNS_ORDER = (
    ["Product Type", "Food Product Group", "Food Product Category", "Product Name"]
    + ["Basic Type", "Sub-Type 1", "Sub-Type 2"]
    + NORMALIZED_COLUMNS
)

df_split = df_split[COLUMNS_ORDER]

df_split.to_csv("better-test-data-cleaning.csv", index=False)

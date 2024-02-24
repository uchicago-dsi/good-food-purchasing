import pandas as pd
import os
from datetime import datetime

from cgfp.config_tags import (
    CATEGORY_TAGS,
    GROUP_TAGS,
    TOKEN_MAP_DICT,
    SKIP_TOKENS,
    FLAVORS,
    CHOCOLATE,
    SHAPE_EXTRAS,
    SKIP_FLAVORS,
)

# TODO: set this up so there's a make command that handles filepaths well
# Right now have to run this from the scripts folder

DATA_FOLDER = "../data/"
RAW_FOLDER = DATA_FOLDER + "raw/"
CLEAN_FOLDER = DATA_FOLDER + "clean/"
RUN_FOLDER = f"pipeline-{datetime.now().strftime('%Y-%m-%d %H-%M')}/"

if not os.path.exists(CLEAN_FOLDER + RUN_FOLDER):
    os.makedirs(CLEAN_FOLDER + RUN_FOLDER)

# TODO: Move this stuff to the config file and figure out the best structure
# for keeping track of the input columns, intermediate steps, and output file
GROUP_COLUMNS = [
    "Product Type",
    "Product Name",
    "Food Product Group",
    "Food Product Category",
    "Primary Food Product Category",
]

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

# TODO: Pretty sure this can be done better
# Need to review the expected input and outputs to clean this up
COLUMNS_ORDER = (
    ["Product Type", "Food Product Group", "Food Product Category", "Product Name"]
    + ["Basic Type"]
    + NORMALIZED_COLUMNS
)


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
    # Remove leading 'PREQUALIFIED: ' string
    df["Product Name"] = df["Product Name"].str.replace(
        "^PREQUALIFIED: ", "", regex=True
    )
    return df


def token_handler(token, food_product_group, food_product_category, basic_type):
    # Handle edge cases where a token is allowed
    if token == "blue" and basic_type == "cheese":
        return token

    if token == "instant" and food_product_group == "Beverages":
        return token

    if token == "black" and basic_type in ["tea", "drink"]:
        return token

    # Handle edge cases where a token is not allowed
    if basic_type == "plant milk" and token in ["nonfat", "low fat"]:
        return None

    if basic_type == "bean" and token == "turtle":
        return None

    if food_product_group == "Milk & Dairy" and token == "in brine":
        return None

    # Map flavored tokens to "flavored" for beverages
    # TODO: Is this right for beverages?
    if food_product_group == "Beverages" and token in FLAVORS:
        return "flavored"

    # Skip flavors and shapes for candy, chips, condiments, etc.
    if basic_type in SKIP_FLAVORS and token in (FLAVORS | SHAPE_EXTRAS):
        return None

    # Map chocolate tokens to "chocolate" for candy
    if basic_type == "candy" and token in CHOCOLATE:
        return "chocolate"

    # Remove wheat from most grain products
    if (
        food_product_category == "Grain Products"
        and basic_type not in ["cereal"]
        and token == "wheat"
    ):
        return None

    # Skip outdated tokens from old name normalization format
    if token in SKIP_TOKENS:
        return None
    return token


def clean_name(
    name_list,
    food_product_group,
    food_product_category,
    group_tags_dict,
    category_tags_dict,
):
    normalized_name = {}
    misc_col = {"Misc": []}  # make a list so we can append unmatched tokens
    for i, token in enumerate(name_list):
        token = token.strip()
        token = TOKEN_MAP_DICT.get(token, token)
        # First token is always Basic Type
        if i == 0:
            basic_type = token
            normalized_name["Basic Type"] = token
            continue
        token = token_handler(
            token, food_product_group, food_product_category, basic_type
        )
        if token is None:
            continue
        # Check if token is in tags — if so, enter the tagging loop
        if token in group_tags_dict.get(food_product_group, {}).get(
            "All", []
        ) or token in category_tags_dict.get(food_product_category, {}).get("All", []):
            matched = False
            for col in NORMALIZED_COLUMNS:
                # TODO: Write better documentation here
                # Find the category that the token is in and add to normalized_name
                if col in group_tags_dict[food_product_group]:
                    if token in group_tags_dict[food_product_group][col]:
                        normalized_name[col] = token
                        matched = True
                        break
            if matched:
                continue
        # First token after basic type is sub-type 1 if it's not from the later tags
        if "Sub-Type 1" not in normalized_name:
            normalized_name["Sub-Type 1"] = token
            continue
        elif "Sub-Type 2" not in normalized_name:
            normalized_name["Sub-Type 2"] = token
            continue
        # Aggregate unmatched tokens to add to tag dictionary
        misc_col["Misc"].append(token)
    normalized_name.update(misc_col)
    # Make sure all columns are represented in dictionary for dataframe creation
    for col in NORMALIZED_COLUMNS:
        if col not in normalized_name:
            normalized_name[col] = None
    return normalized_name


def pool_tags(tags_dict):
    for top_level in tags_dict.keys():
        tags_dict[top_level]["All"] = set.union(*tags_dict[top_level].values())
    return tags_dict


if __name__ == "__main__":
    # TODO: Add args for filepath, etc.
    INPUT_FILE = "CONFIDENTIAL_CGFP bulk data_073123.csv"
    MISC_FILE = "misc.csv"
    CLEAN_FILE = "clean_" + INPUT_FILE

    INPUT_PATH = RAW_FOLDER + INPUT_FILE
    MISC_PATH = CLEAN_FOLDER + RUN_FOLDER + MISC_FILE
    CLEAN_PATH = CLEAN_FOLDER + RUN_FOLDER + CLEAN_FILE

    df = pd.read_csv(INPUT_PATH)
    df["Misc"] = None
    df = clean_df(df)

    group_tags = pool_tags(GROUP_TAGS)
    category_tags = pool_tags(CATEGORY_TAGS)

    # Get a dictionary back with split tags allocated to appropriate columns
    # Then convert dictionary to dataframe
    name_dict = df.apply(
        lambda row: clean_name(
            row["Product Name"].split(","),
            row["Food Product Group"],
            row["Food Product Category"],
            group_tags,
            category_tags,
        ),
        axis=1,
    )
    new_columns = pd.DataFrame(name_dict.tolist()).reset_index(drop=True)

    # Combine split tags with group and category columns
    df_split = pd.concat(
        [
            df[GROUP_COLUMNS],
            new_columns,
        ],
        axis=1,
    )

    # TODO: Handle sub-type 3 when we add that
    # Replace multiple flavors for juices with "blend"
    condition = (
        (df_split["Basic Type"] == "juice")
        & (df_split["Sub-Type 1"].isin(FLAVORS))
        & (df_split["Sub-Type 2"].isin(FLAVORS))
    )

    df_split[condition]["Sub Type 1"] = "blend"
    df_split[condition]["Sub Type 2"] = None

    # Save unallocated tags for manual review
    misc = df_split[df_split["Misc"].apply(lambda x: x != [])][
        [
            "Product Type",
            "Food Product Group",
            "Food Product Category",
            "Basic Type",
            "Sub-Type 1",
            "Sub-Type 2",
            "Misc",
        ]
    ]

    MISC_SORT_ORDER = [
        "Food Product Group",
        "Food Product Category",
        "Basic Type",
    ]
    misc = misc.sort_values(by=MISC_SORT_ORDER)
    misc.to_csv(MISC_PATH, index=False)

    TAGS_SORT_ORDER = [
        "Food Product Group",
        "Food Product Category",
        "Basic Type",
        "Sub-Type 1",
        "Sub-Type 2",
    ]

    df_split = df_split[COLUMNS_ORDER].sort_values(by=TAGS_SORT_ORDER)
    df_split.to_csv(CLEAN_PATH, index=False)

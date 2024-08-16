"""Data cleaning pipeline for raw CGFP data"""

import argparse
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import pandas as pd
from ordered_set import OrderedSet
from tqdm import tqdm

from cgfp.constants.pipeline_constants import (
    ADDITIONAL_COLUMNS,
    COLUMNS_ORDER,
    GROUP_COLUMNS,
    NON_SUBTYPE_COLUMNS,
    NORMALIZED_COLUMNS,
    RUN_FOLDER,
    SUBTYPE_COLUMNS,
)
from cgfp.constants.tokens.basic_type_mapping import BASIC_TYPE_MAPPING
from cgfp.constants.tokens.misc_tags import NON_SUBTYPE_TAGS_FPC
from cgfp.constants.tokens.product_type_mapping import PRODUCT_TYPE_MAPPING
from cgfp.constants.tokens.skip_tokens import SKIP_TOKENS
from cgfp.constants.tokens.tag_sets import (
    ALL_FLAVORS,
    CHEESE_TYPES,
    CHOCOLATE,
    CORN_CERAL,
    FLAVORED_BASIC_TYPES,
    FLAVORS,
    FRUIT_SNACKS,
    FRUITS,
    MELON_TYPES,
    NUTS,
    OAT_CEREAL,
    SHAPE_EXTRAS,
    SKIP_FLAVORS,
    SKIP_SHAPE,
    SUBTYPE_REPLACEMENT_MAPPING,
    VEGETABLES,
    WHEAT_CEREAL,
)
from cgfp.constants.tokens.token_map import TOKEN_MAP_DICT
from cgfp.util import load_to_pd, save_pd_to_csv

tqdm.pandas()

DEFAULT_INPUT_FILE = "CONFIDENTIAL_CGFP bulk data_073123.xlsx"
DEFAULT_MISC_FILE = "misc.csv"
CLEAN_FILE_PREFIX = "clean_"

# TODO: Set up a config or something
SMOKE_TEST = False


def create_parser() -> argparse.ArgumentParser:
    """Creates and returns an argument parser for processing files.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument("--input_file", default=DEFAULT_INPUT_FILE, help="Input file path")
    parser.add_argument("--clean_folder", default="./data/clean/", help="")
    parser.add_argument(
        "--clean_file",
        default=None,
        help="Clean file path. If not specified, it will be automatically generated based on the input file.",
    )
    parser.add_argument("--misc_file", default=DEFAULT_MISC_FILE, help="Miscellaneous file path")
    parser.add_argument("--raw_data", default="./data/raw/", help="Relative path to raw data directory")
    parser.add_argument("--disable-output", action="store_false", dest="do_write_output", default=True)
    return parser


def clean_df(df_cgfp: pd.DataFrame, str_len_threshold: int = 3) -> pd.DataFrame:
    """Cleans the given DataFrame by applying several filters and transformations:

    Args:
        df_cgfp: The DataFrame to clean.
        str_len_threshold: The minimum length for "Product Type" and "Product Name"

    Returns:
        The cleaned DataFrame.
    """
    # TODO: Do we ever use "Primary Food Product Group?
    df_cgfp = df_cgfp[ADDITIONAL_COLUMNS + GROUP_COLUMNS].copy()

    # Add normalized name columns
    df_cgfp[NORMALIZED_COLUMNS + ["Misc"]] = None

    df_cgfp = df_cgfp[
        (df_cgfp["Product Type"].str.len() >= str_len_threshold)
        & (df_cgfp["Product Name"].str.len() >= str_len_threshold)
        & (df_cgfp["Food Product Group"] != "Non-Food")
    ].reset_index(drop=True)

    # Handle typos in Primary Food Product Category
    category_typos = {
        "Roots & Tuber": "Roots & Tubers",
    }
    df_cgfp["Primary Food Product Category"] = df_cgfp["Primary Food Product Category"].map(
        lambda x: category_typos.get(x, x)
    )

    # Replace "Whole/Minimally Processed" with the value from "Food Product Category"
    df_cgfp["Primary Food Product Category"] = df_cgfp.progress_apply(
        lambda row: (
            row["Food Product Category"]
            if row["Primary Food Product Category"] == "Whole/Minimally Processed"
            else row["Primary Food Product Category"]
        ),
        axis=1,
    )

    # Remove leading 'PREQUALIFIED: ' string
    df_cgfp["Product Name"] = df_cgfp["Product Name"].str.replace("^PREQUALIFIED: ", "", regex=True)
    return df_cgfp


def token_handler(token: str, row: pd.Series) -> Tuple[Optional[str], pd.Series]:
    """Handles token processing for specific edge cases in a data row.

    Args:
        token: The token to be processed.
        row: The data row represented as a pandas Series.

    Returns:
        A tuple where the first element is either the processed token or None, and the second element is the potentially modified row.
    """
    food_product_group, food_product_category, basic_type, sub_type_1 = (
        row["Food Product Group"],
        row["Food Product Category"],
        row["Basic Type"],
        row["Sub-Type 1"],
    )

    # Handle edge cases where Basic Type should change to Sub-Type
    if (basic_type == "snack" and token == "bar") or (basic_type == "herb" and token == "watercress"):
        row["Basic Type"] = token
        return None, row

    # Handle edge cases where a token is allowed
    if (
        (token == "blue" and basic_type == "cheese")
        or (token == "instant" and food_product_group == "Beverages")
        or (token == "black" and basic_type in ["tea", "drink"])
        or (token == "gluten free" and sub_type_1 in ["parfait"])
    ):
        return token, row

    # Handle edge cases where a token is not allowed
    if (
        # Food product group rules
        (food_product_group == "Milk & Dairy" and token in ["in brine", "nectar", "honey"])
        or (food_product_group == "Meat" and token in ["ketchup", "italian"])
        or (food_product_group == "Produce" and token in ["whole", "peeled", "kosher", "gluten free"])
        or (food_product_group == "Seafood" and token in ["seasoned", "stuffed", "lime"])
        or (
            food_product_group == "Condiments & Snacks"
            and token in ["shredded", "non-dairy"]
            # Food product category rules
            or (
                food_product_category == "Cheese"
                and token
                in [
                    "in water",
                    "ball",
                    "low moisture",
                    "whole milk",
                    "logs",
                    "unsalted",
                    "in oil",
                ]
            )
            # Basic type rules
            or (basic_type == "plant milk" and token in ["nonfat", "low fat"])
            or (basic_type == "bean" and token == "turtle")
            or (basic_type == "supplement" and token == "liquid")
            or (basic_type == "bar" and token in ["cereal", "cocoa", "seed"])
            or (basic_type == "ice cream" and token in ["crunch", "taco", "chocolate covered", "cookie"])
            or (basic_type == "salsa" and token in ["thick", "chunky", "mild"])
            or (food_product_category == "Grain Products" and basic_type not in ["cereal"] and token == "wheat")
            or (basic_type == "condiment" and token in ["thick", "thin", "sweet"])
            or (basic_type == "cookie" and token in ["sugar"])
            or (basic_type == "dessert" and token in ["crumb", "graham cracker"])
            or (basic_type == "mix" and token in ["custard"])
        )
    ):
        return None, row

    # Map flavored tokens to "flavored"
    if (
        (
            (
                food_product_group == "Beverages"
                or food_product_category == "Cheese"
                or basic_type in FLAVORED_BASIC_TYPES
            )
            and token in ALL_FLAVORS
        )
        or (basic_type == "bar" and token in FLAVORS)  # TODO: this doesn't work since sub-type 1 is bar
        or (food_product_group == "Seafood" and token in FLAVORS)
    ):
        return "flavored", row

    # Skip flavors and shapes for some basic types
    if basic_type in SKIP_FLAVORS and token in ALL_FLAVORS:
        return None, row

    if basic_type in SKIP_SHAPE and token in SHAPE_EXTRAS:
        return None, row

    # EDGE CASES FOR NON-SUBTYPE COLUMNS #

    if token == "grated" and food_product_category != "Cheese":
        return "cut", row

    if token == "mix" and food_product_group == "Beverages":
        return "concentrate", row

    if token == "taco meat":
        row["Shape"] = "crumble"
        row["Processing"] = "seasoned"
        return None, row

    if token == "pulled" and food_product_group in ["Meat", "Meals"]:
        row["Shape"] = "cut"
        row["Cooked/Cleaned"] = "cooked"
        return None, row
    elif token == "pulled":
        return None, row

    if token == "stick" and food_product_group in ["Produce", "Seafood"]:
        return "cut", row

    if token == "stick" and food_product_category in ["Cheese", "Meat"]:
        return "ss", row

    if token == "shredded" and food_product_group == "Meat":
        row["Shape"] = "cut"
        row["Cooked/Cleaned"] = "cooked"
        return None, row
    elif token == "shredded":
        return "cut", row

    if token == "powder" and food_product_group == "Beverages":
        return "concentrate", row

    if token == "popper":
        row = add_subtypes(row, "cheese")
        row["Processing"] = "breaded"
        return None, row

    if token == "popcorn" and food_product_category in ["Seafood", "Chicken"]:
        row["Shape"] = "cut"
        row["Processing"] = "breaded"
        return None, row

    # EDGE CASES FOR RENAMING TOKENS #

    # Map nut tokens to "nut" for some basic types
    if basic_type == "snack" and token in NUTS:
        return "nut", row

    # Relabel cheese type as "cheese"
    if (food_product_group == "Meals" or basic_type == "snack") and token in CHEESE_TYPES:
        return "cheese", row

    # Map chocolate tokens to "chocolate" for candy
    if basic_type == "candy" and token in CHOCOLATE:
        return "chocolate", row

    # "chip" should be mapped to "cut" for pickles...but "chip" is valid for snacks
    if sub_type_1 == "pickle" and token == "chip":
        return "cut", row

    if token == "base" and food_product_group == "Beverages":
        return "mix", row

    if token == "grated" and food_product_group != "Milk & Dairy":
        return "cut", row

    if token == "string" and basic_type == "cheese":
        return "ss", row

    # Skip outdated tokens from old name normalization format
    # Do this last since some rules override this
    if token in SKIP_TOKENS:
        return None, row
    return token, row


def clean_token(token: str, token_map_dict: dict = TOKEN_MAP_DICT) -> str:
    """Cleans a token maps it to a corrected value

    Args:
        token: The token to be cleaned
        token_map_dict: A dictionary that maps incorrect tokens to their correct values

    Returns:
        The cleaned and mapped token
    """
    cleaned_token = token.strip().lower()
    # can have multiple mappings — may be a typo that maps to another typo that is then mapped to the correct token
    while cleaned_token in token_map_dict:
        cleaned_token = token_map_dict[cleaned_token]
    return cleaned_token


def basic_type_handler(row):
    """Handles the processing of a row based on the "Basic Type" mapping.

    Args:
        row: The row to be processed.

    Returns:
        The modified row with updates based on the "Basic Type" mapping.
    """
    mapping = BASIC_TYPE_MAPPING.get(row["Basic Type"], None)

    if mapping is None:
        return row

    # Note: This assigns given values to these columns without changing other ones
    for key, value in mapping.items():
        if key != "Sub-Types":
            row[key] = value

    # TODO: Ok yeah need to update the row and also the subtypes
    # Maybe make a separate "update_subtypes" function here?
    # Or is this ok?
    if "Sub-Types" in mapping:
        row = add_subtypes(row, mapping["Sub-Types"], first=True)
    return row


def add_subtypes(row: dict, tokens: Union[str, List[str]], first: bool = False) -> dict:
    """Adds subtypes to the "Sub-Types" field in the row, with an option to prioritize new tokens.

    Args:
        row: The row to update with subtypes.
        tokens: The token or list of tokens to add as subtypes.
        first: If True, adds the new tokens first before existing subtypes.

    Returns:
        The updated row with the modified "Sub-Types" field.
    """
    # Ensure tokens is a list
    if not isinstance(tokens, list):
        tokens = [tokens]

    # Note: Reorder the set with the new token(s) first if 'first' is True
    if first:
        subtypes = OrderedSet(tokens)
        subtypes.update(row.get("Sub-Types", []))
        row["Sub-Types"] = subtypes
    else:
        subtypes = OrderedSet(row.get("Sub-Types", []))
        subtypes.update(tokens)
        row["Sub-Types"] = subtypes

    row = update_subtypes(row)
    return row


def remove_subtypes(row: dict, tokens: Union[str, List[str]]) -> dict:
    """Removes specified subtypes from the "Sub-Types" field in the row.

    Args:
        row: The row to update.
        tokens: The token or list of tokens to remove from the subtypes.

    Returns:
        The updated row with the specified subtypes removed.
    """
    if not isinstance(tokens, list):
        tokens = [tokens]

    subtypes = OrderedSet(row.get("Sub-Types", []))

    for token in tokens:
        subtypes.discard(token)

    row["Sub-Types"] = subtypes
    row = update_subtypes(row)
    return row


def update_subtypes(row: dict, num_subtype_cols: int = 2) -> dict:
    """Updates row with subtypes in individual columns

    Args:
        row: The row to update.
        num_subtype_cols: The number of subtype columns

    Returns:
        The updated row with subtypes distributed across the appropriate columns.
    """
    # TODO: maybe you can zip this with the sub-type columns?
    for i, subtype in enumerate(row["Sub-Types"]):
        if i == 0:
            row["Sub-Type 1"] = subtype
        elif i == 1:
            row["Sub-Type 2"] = subtype
        else:
            # Not enough room!
            break
    row["Misc"] = list(row["Sub-Types"])[num_subtype_cols:] if len(row["Sub-Types"]) > num_subtype_cols else []
    return row


# TODO: Set this up like basic_type_handler with a mapping dictionary
def subtype_handler(row: dict, token: str) -> Tuple[Optional[str], dict]:
    """Handles specific token cases to update the row's attributes based on predefined rules.

    Args:
        row: The row to update.
        token: The token to process.

    Returns:
        A tuple where the first element is either a processed token or None, and the second element is the updated row.
    """
    if token == "2% lactose free":
        row["Dietary Accommodation"] = "lactose free"
        row["Dietary Concern"] = "2%"
        return None, row

    if token == "apple juice":
        row["Basic Type"] = "juice"
        return "apple", row

    if token == "applesauce" and row["Basic Type"] != "baby food":
        return None, row

    if token == "cheez-it":
        row["Basic Type"] = "cracker"
        return "cheese", row

    if token == "earl grey" and row["Food Product Category"] != "Beverages":
        return "flavored", row

    # TODO: Maybe move the other subtype rules here?
    if token == "french toast bread":
        row["Basic Type"] = "french toast"
        return None, row

    if token == "fried onion":
        row["Basic Type"] = "topping"
        # TODO: Wait should "fried" be in one of the processing cols?
        row = add_subtypes(row, ["onion", "fried"], first=True)
        return None, row

    if token == "fruit and vegetable" and row["Food Product Group"] == "Beverages":
        return "fruit punch", row

    if token == "fruit medley" and row["Basic Type"] == "juice":
        return "blend", row

    if token == "fruit bar":
        row["Basic Type"] = "popsicle"
        return "fruit", row

    if token == "funnel cake" and row["Basic Type"] == "dessert":
        return "cake", row

    if token == "gherkin":
        row["Basic Type"] = "condiment"
        return "pickle", row

    if token == "gravy master":
        row["Basic Type"] = "sauce"
        return "browning", row

    if token in FRUIT_SNACKS:
        row["Basic Type"] = "fruit snack"
        return None, row

    # Note: these all have "cereal" as basic type so convert subtype to grain type
    if token in WHEAT_CEREAL:
        return "wheat", row

    if token in CORN_CERAL:
        return "corn", row

    if token in OAT_CEREAL:
        return "oat", row

    return token, row


def postprocess_subtypes(row: dict) -> dict:
    """Applies postprocessing rules to the subtypes in the row.

    Args:
        row: The row to process.

    Returns:
        The processed row
    """
    # TODO: Make this robust to subtype changes, change to subtype 3, etc.
    # Count occurrences of each category
    category_counts = {}
    for subtype in SUBTYPE_COLUMNS:
        category = get_category(row[subtype])
        if category:
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts[category] = 1

    # TODO: this works with up to three subtypes — could break with more
    # Replace subtypes if more than one belongs to the same category
    for category, count in category_counts.items():
        if count > 1:
            if category == "fruit" and row["Food Product Category"] == "Fruit":
                replacement_value = "blend"
            else:
                replacement_value = SUBTYPE_REPLACEMENT_MAPPING.get(category)

            replaced = False
            for subtype in SUBTYPE_COLUMNS:
                if get_category(row[subtype]) == category:
                    row["Sub-Types"].discard(row[subtype])
                    if not replaced:
                        row["Sub-Types"].add(replacement_value)
                        row[subtype] = replacement_value
                        replaced = True
                    else:
                        row[subtype] = None
    row = update_subtypes(row)
    return row


def clean_name(row: dict) -> dict:
    """Cleans and processes the "Product Name" in the row by handling edge cases, assigning tags, and updating subtypes based on predefined rules.

    Args:
        row: The row to clean and process.

    Returns:
        The updated row with cleaned and normalized values.
    """
    # Note: Need to add "Sub-Types" to the row first thing
    row["Sub-Types"] = OrderedSet()

    # Handle product type edge cases — short-circuit if a mapping exists
    if row["Product Type"] in PRODUCT_TYPE_MAPPING:
        mapping = PRODUCT_TYPE_MAPPING[row["Product Type"]]
        for key, value in mapping.items():
            if key != "Sub-Types":
                row[key] = value
        subtypes = mapping.get("Sub-Types", [])
        row = add_subtypes(row, subtypes)
        row = update_subtypes(row)
        return row

    food_product_category = row["Food Product Category"]
    # Tags are allowed based on primary food product category for meals
    if food_product_category == "Meals":
        food_product_category = row["Primary Food Product Category"]
    product_name_split = row["Product Name"].split(",")
    row["Misc"] = []

    basic_type = clean_token(product_name_split[0])
    row["Basic Type"] = basic_type
    row = basic_type_handler(row)

    for token in product_name_split[1:]:
        token = clean_token(token)
        token, row = token_handler(token, row)
        if token is None:
            continue  # token_handler returns None for invalid tags so skip
        # If token is allowed in a non-subtype column, put it there
        # Otherwise, add to subtypes
        if token in NON_SUBTYPE_TAGS_FPC.get(food_product_category, {}).get("All", []):
            matched = False
            # Note: Skip "Basic Type" column since it's already set
            for col in NON_SUBTYPE_COLUMNS:
                if token in NON_SUBTYPE_TAGS_FPC[food_product_category][col]:
                    # Duplicate entry for column, add to subtypes
                    if row[col] is not None:
                        break
                    row[col] = token
                    matched = True
                    break
            if matched:
                continue
        # Unmatched tokens are subtypes
        token, row = subtype_handler(row, token)  # handles subtype edge cases
        if token is not None:
            row = add_subtypes(row, token)

    # Handle edge cases not captured by other rules
    row = postprocess_data(row)
    # Apply subtype rules for specific groups and categories
    row = postprocess_subtypes(row)

    # Deduplicate column values
    row_normalized = row[NORMALIZED_COLUMNS]
    row_normalized[row_normalized.notna() & row_normalized.duplicated()] = None
    row[NORMALIZED_COLUMNS] = row_normalized
    return row


def get_category(subtype: str) -> Optional[str]:
    """Determines the category of a given subtype.

    Args:
        subtype: The subtype to categorize.

    Returns:
        The category as a string if the subtype matches a known category, otherwise None.
    """
    # Helper function to determine the category of a subtype
    if subtype in FRUITS:
        return "fruit"
    elif subtype in CHEESE_TYPES:
        return "cheese"
    elif subtype in VEGETABLES:
        return "vegetable"
    elif subtype in MELON_TYPES:
        return "melon"
    return None


def clear_row(row: dict) -> dict:
    """Clears specified fields in the row, except for "Basic Type".

    Args:
        row: The row to clear.

    Returns:
        The cleared row with specified fields set to None, and "Sub-Types" and "Misc" reset.
    """
    # Note: Don't clear Basic Type
    for col in NORMALIZED_COLUMNS[1:]:
        row[col] = None
    row["Sub-Types"] = OrderedSet()
    row["Misc"] = []
    return row


def postprocess_data(row: dict) -> dict:
    """Applies postprocessing rules to the row

    Args:
        row: The row to process for edge cases.

    Returns:
        The updated row with corrected values based on the edge case rules.
    """
    ### Handle edge cases for mislabeled data ###
    # "spice" is always "Condiments & Snacks"
    if row["Basic Type"] == "spice" and row["Food Product Group"] != "Condiments & Snacks":
        row["Food Product Group"] = "Condiments & Snacks"
        row["Food Product Category"] = "Condiments & Snacks"
        row["Primary Product Category"] = "Condiments & Snacks"

    # Handle "chili" as Basic Type
    if row["Basic Type"] == "chili" and row["Food Product Group"] == "Condiments & Snacks":
        row["Basic Type"] = "spice"
        row = add_subtypes(row, "chili", first=True)
        return row
    if row["Basic Type"] == "chili" and row["Food Product Group"] == "Produce":
        row["Basic Type"] = "pepper"
        row = add_subtypes(row, "chili", first=True)
        return row

    # Assume that bologna is made with beef, pork, and chicken so label with "Beef" as
    # Food Product Category since that has highest climate impact
    if row["Basic Type"] == "bologna" and "all" in row["Product Type"].lower():
        row["Basic Type"] = "beef"
        row = remove_subtypes(row, list(row["Sub-Types"]))
        row = add_subtypes(row, ["pork", "bologna"])
        row["Food Product Category"] = "Beef"
        row["Primary Food Product Category"] = "Beef"
        return row

    # Roasted chickpeas are Basic Type "snack"
    # TODO: Should FPC always be Condiments & Snacks? then?
    if row["Basic Type"] == "chickpea" and "roasted" in row["Product Name"].lower():
        row["Basic Type"] = "snack"
        row = add_subtypes(row, "chickpea", first=True)
        return row

    if row["Basic Type"] == "beverage" and row["Sub-Type 1"] == "energy drink":
        row["Basic Type"] = "energy drink"
        # Note: Subtypes are finicky so we need to actually remove them with the remove_subtypes function
        row = remove_subtypes(row, "energy drink")
        return row

    return row


# TODO: Set up smoke test in config
def process_data(df_cgfp, smoke_test=SMOKE_TEST, **options):
    """Processes the given DataFrame by filtering, cleaning, normalizing names, and creating a diff file.

    Args:
        df_cgfp: The DataFrame to process.
        smoke_test: If True, limits the DataFrame to the first 1000 rows for testing.
        **options: Additional keyword arguments that can be passed to customize the processing.

    Returns:
        The processed DataFrame with normalized names and other adjustments.
    """
    if smoke_test:
        df_cgfp = df_cgfp.head(1000)

    # Filter missing data and non-food items, handle typos in Category and Group columns
    df_cgfp = clean_df(df_cgfp)

    # Create normalized name
    print("Normalizing names...")
    df_normalized = df_cgfp.progress_apply(clean_name, axis=1)

    # Perform diff on "Normalized Name" column with "Product Name" column from df_loaded
    # Save a diff on the "Product Name" column with the edited output
    print("Creating diff file...")
    df_normalized["Normalized Name"] = df_normalized.progress_apply(
        lambda row: ", ".join(row[NORMALIZED_COLUMNS].dropna().astype(str)),
        axis=1,
    )
    df_normalized["Sub-Types"] = df_normalized["Sub-Types"].apply(lambda x: str(list(x)))

    df_diff = df_cgfp["Product Name"].compare(df_normalized["Normalized Name"])
    df_diff["Product Type"] = df_cgfp["Product Type"]
    df_diff = df_diff[["Product Type"] + [col for col in df_diff.columns if col != "Product Type"]]
    df_diff = df_diff.sort_values(by="self")

    # Reset index for future sorting
    df_normalized = df_normalized.reset_index(drop=True)

    # If there are more subtype tags than allowed in the subtype columns, they are saved here for review
    print("Creating misc file...")
    misc = df_normalized[df_normalized["Misc"].progress_apply(lambda x: x != [])][
        [
            "Product Type",
            "Product Name",
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

    TAGS_SORT_ORDER = [
        "Food Product Group",
        "Food Product Category",
        "Basic Type",
        "Sub-Type 1",
        "Sub-Type 2",
    ]

    df_scoring = df_normalized.drop(columns=["Product Name"]).rename(columns={"Normalized Name": "Product Name"})
    df_scoring = df_scoring[ADDITIONAL_COLUMNS + COLUMNS_ORDER]

    df_normalized = df_normalized[COLUMNS_ORDER + ["Sub-Types"]].sort_values(by=TAGS_SORT_ORDER)

    # return processed assets to main
    return df_normalized, misc, df_diff, df_scoring


def main(argv):
    """Main function to handle data loading, processing, and saving.

    Args:
        argv: List of command-line arguments.

    Returns:
        None
    """
    # input
    parser = create_parser()
    # TODO: wait what is this doing?
    options = vars(parser.parse_args(argv))

    # processing
    print("Loading data...")
    df_loaded = load_to_pd(**options)
    df_processed, misc, df_diff, df_scoring = process_data(df_loaded, **options)

    # output
    # TODO: I...don't get this
    print("Saving files...")
    save_pd_to_csv(df_processed, **options)
    save_pd_to_csv(
        misc,
        options.get("clean_folder"),
        options.get("misc_file"),
        output_file="misc.csv",
    )

    # Save file for new scoring platform
    scoring_file = RUN_FOLDER / "scoring.csv"
    df_scoring.to_csv(scoring_file, index=False)

    # Save diff file
    diff_file = RUN_FOLDER / "normalized_name_diff.csv"
    df_diff.to_csv(diff_file, index=False)

    # Combine counts for each column
    counts_dict = {}
    for col in df_processed.columns:
        counts_dict[col] = df_processed[col].value_counts()

    # Combine subtype counts
    combined_subtype_counts = defaultdict(int)
    for col in SUBTYPE_COLUMNS:
        if col in counts_dict:
            for value, count in counts_dict[col].items():
                combined_subtype_counts[value] += count
        del counts_dict[col]

    counts_dict["Sub-Types"] = pd.Series(
        dict(sorted(combined_subtype_counts.items(), key=lambda item: item[1], reverse=True))
    )

    sorted_counts_dict = {}
    for column in counts_dict.keys():
        if column == "Basic Type":
            sorted_counts_dict[column] = counts_dict[column]
            if "Sub-Types" in counts_dict:
                sorted_counts_dict["Sub-Types"] = counts_dict["Sub-Types"]
        elif column != "Sub-Types":
            sorted_counts_dict[column] = counts_dict[column]

    counts_file = RUN_FOLDER / "value_counts.xlsx"

    # Write the counts to an Excel file
    with pd.ExcelWriter(counts_file) as writer:
        for column, counts in sorted_counts_dict.items():
            df_counts = counts.reset_index()
            df_counts.columns = [column, "Count"]
            df_counts.to_excel(writer, sheet_name=column.replace("/", "_"), index=False)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])

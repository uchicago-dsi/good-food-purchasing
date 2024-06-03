import pandas as pd
import argparse
from ordered_set import OrderedSet
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from cgfp.constants.tag_sets import (
    FLAVORS,
    FRUITS,
    CHOCOLATE,
    SHAPE_EXTRAS,
    SKIP_FLAVORS,
    FLAVORED_BASIC_TYPES,
    NUTS,
    CHEESE_TYPES,
    VEGETABLES,
    MELON_TYPES,
    SKIP_SHAPE,
    ALL_FLAVORS,
    SUBTYPE_REPLACEMENT_MAPPING,
)
from cgfp.constants.misc_tags import NON_SUBTYPE_TAGS_FPC
from cgfp.constants.token_map import TOKEN_MAP_DICT
from cgfp.constants.skip_tokens import SKIP_TOKENS
from cgfp.constants.product_type_mapping import PRODUCT_TYPE_MAPPING
from cgfp.constants.pipeline import (
    CLEAN_FOLDER,
    RUN_FOLDER,
    GROUP_COLUMNS,
    SUBTYPE_COLUMNS,
    NON_SUBTYPE_COLUMNS,
    NORMALIZED_COLUMNS,
    COLUMNS_ORDER,
)
from cgfp.constants.basic_type_mapping import BASIC_TYPE_MAPPING
from cgfp.util import load_to_pd, save_pd_to_csv

tqdm.pandas()

DEFAULT_INPUT_FILE = "CONFIDENTIAL_CGFP bulk data_073123.xlsx"
DEFAULT_MISC_FILE = "misc.csv"
CLEAN_FILE_PREFIX = "clean_"


def create_parser():
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument(
        "--input_file", default=DEFAULT_INPUT_FILE, help="Input file path"
    )
    parser.add_argument("--clean_folder", default="./data/clean/", help="")
    parser.add_argument(
        "--clean_file",
        default=None,
        help="Clean file path. If not specified, it will be automatically generated based on the input file.",
    )
    parser.add_argument(
        "--misc_file", default=DEFAULT_MISC_FILE, help="Miscellaneous file path"
    )
    parser.add_argument(
        "--raw_data", default="./data/raw/", help="Relative path to raw data directory"
    )
    parser.add_argument(
        "--disable-output", action="store_false", dest="do_write_output", default=True
    )
    return parser


def clean_df(df):
    """
    Cleaning:
    - Remove null and short (usually a mistake) Product Types
    - Remove null and short (usually a mistake) Product Names
    - Remove non-food items
    """
    # TODO: Do we ever use "Primary Food Product Group?
    df = df[GROUP_COLUMNS].copy()

    # Add normalized name columns
    df[NORMALIZED_COLUMNS + ["Misc"]] = None

    df = df[
        (df["Product Type"].str.len() >= 3)
        & (df["Product Name"].str.len() >= 3)
        & (df["Food Product Group"] != "Non-Food")
    ].reset_index(drop=True)

    # Handle typos in Primary Food Product Category
    category_typos = {
        "Roots & Tuber": "Roots & Tubers",
    }
    df["Primary Food Product Category"] = df["Primary Food Product Category"].map(
        lambda x: category_typos.get(x, x)
    )

    # Replace "Whole/Minimally Processed" with the value from "Food Product Category"
    df["Primary Food Product Category"] = df.progress_apply(
        lambda row: (
            row["Food Product Category"]
            if row["Primary Food Product Category"] == "Whole/Minimally Processed"
            else row["Primary Food Product Category"]
        ),
        axis=1,
    )

    # Remove leading 'PREQUALIFIED: ' string
    df["Product Name"] = df["Product Name"].str.replace(
        "^PREQUALIFIED: ", "", regex=True
    )
    return df


def token_handler(token, row):
    food_product_group, food_product_category, basic_type, sub_type_1 = (
        row["Food Product Group"],
        row["Food Product Category"],
        row["Basic Type"],
        row["Sub-Type 1"],
    )

    # Handle edge cases where Basic Type should change to Sub-Type
    if (basic_type == "snack" and token == "bar") or (
        basic_type == "herb" and token == "watercress"
    ):
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
        (
            food_product_group == "Milk & Dairy"
            and token in ["in brine", "nectar", "honey"]
        )
        or (food_product_group == "Meat" and token in ["ketchup", "italian"])
        or (
            food_product_group == "Produce"
            and token in ["whole", "peeled", "kosher", "gluten free"]
        )
        or (
            food_product_group == "Seafood" and token in ["seasoned", "stuffed", "lime"]
        )
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
            or (
                basic_type == "ice cream"
                and token in ["crunch", "taco", "chocolate covered", "cookie"]
            )
            or (basic_type == "salsa" and token in ["thick", "chunky", "mild"])
            or (
                food_product_category == "Grain Products"
                and basic_type not in ["cereal"]
                and token == "wheat"
            )
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
        or (
            basic_type == "bar" and token in FLAVORS
        )  # TODO: this doesn't work since sub-type 1 is bar
        or (food_product_group == "Seafood" and token in FLAVORS)
    ):
        return "flavored", row

    # Skip flavors and shapes for some basic types
    if basic_type in SKIP_FLAVORS and token in ALL_FLAVORS:
        return None, row

    if basic_type in SKIP_SHAPE and token in SHAPE_EXTRAS:
        return None, row

    ### EDGE CASES FOR NON-SUBTYPE COLUMNS ###

    if token == "grated" and food_product_category != "Cheese":
        return "cut", row

    if token == "mix" and food_product_group == "Beverages":
        return "concentrate", row

    if token == "taco meat":
        row["Shape"] = "crumble"
        row["Processing"] = "seasoned"
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

    ### EDGE CASES FOR RENAMING TOKENS ###

    # Map nut tokens to "nut" for some basic types
    if basic_type == "snack" and token in NUTS:
        return "nut", row

    # Relabel cheese type as "cheese"
    if (
        food_product_group == "Meals" or basic_type == "snack"
    ) and token in CHEESE_TYPES:
        return "cheese", row

    # Map chocolate tokens to "chocolate" for candy
    if basic_type == "candy" and token in CHOCOLATE:
        return "chocolate", row

    # "chip" should be mapped to "cut" for pickles...but "chip" is valid for snacks
    if sub_type_1 == "pickle" and token == "chip":
        return "cut", row

    # Skip outdated tokens from old name normalization format
    # Do this last since some rules override this
    if token in SKIP_TOKENS:
        return None, row
    return token, row


def clean_token(token, token_map_dict=TOKEN_MAP_DICT):
    cleaned_token = token.strip().lower()
    # can have multiple mappings
    while cleaned_token in token_map_dict:
        cleaned_token = token_map_dict[cleaned_token]
    return cleaned_token


def basic_type_handler(row):
    mapping = BASIC_TYPE_MAPPING.get(row["Basic Type"], None)

    if mapping is None:
        return row

    for key, value in mapping.items():
        if key != "Sub-Types":
            row[key] = value

    # TODO: Ok yeah need to update the row and also the subtypes
    # Maybe make a separate "update_subtypes" function here?
    # Or is this ok?
    if "Sub-Types" in mapping:
        row = add_subtypes(row, mapping["Sub-Types"], first=True)
    return row


def add_subtypes(row, tokens, first=False):
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


def remove_subtypes(row, tokens):
    if not isinstance(tokens, list):
        tokens = [tokens]

    subtypes = OrderedSet(row.get("Sub-Types", []))

    for token in tokens:
        subtypes.discard(token)

    row["Sub-Types"] = subtypes
    row = update_subtypes(row)
    return row


def update_subtypes(row):
    # TODO: maybe you can zip this with the sub-type columns?
    for i, subtype in enumerate(row["Sub-Types"]):
        if i == 0:
            row["Sub-Type 1"] = subtype
        elif i == 1:
            row["Sub-Type 2"] = subtype
        else:
            # Not enough room!
            break
    row["Misc"] = list(row["Sub-Types"])[2:] if len(row["Sub-Types"]) > 2 else []
    return row


def handle_subtypes(row):
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


def clean_name(row):
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
        food_product_category == row["Primary Food Product Category"]
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
        if token in NON_SUBTYPE_TAGS_FPC[food_product_category]["All"]:
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
        row = add_subtypes(row, token)  # Unmatched tokens are subtypes

    # Apply subtype rules for specific groups and categories
    row = handle_subtypes(row)
    # Handle edge cases not captured by other rules
    row = postprocess_data(row)

    # Deduplicate column values
    row_normalized = row[NORMALIZED_COLUMNS]
    row_normalized[row_normalized.notna() & row_normalized.duplicated()] = None
    row[NORMALIZED_COLUMNS] = row_normalized
    return row


def get_category(subtype):
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


def clear_row(row):
    # Note: Don't clear Basic Type
    for col in NORMALIZED_COLUMNS[1:]:
        row[col] = None
    row["Sub-Types"] = OrderedSet()
    row["Misc"] = []
    return row


def postprocess_data(row):
    ### Handle edge cases for mislabeled data ###
    # "spice" is always "Condiments & Snacks"
    if (
        row["Basic Type"] == "spice"
        and row["Food Product Group"] != "Condiments & Snacks"
    ):
        row["Food Product Group"] = "Condiments & Snacks"
        row["Food Product Category"] = "Condiments & Snacks"
        row["Primary Product Category"] = "Condiments & Snacks"

    if row["Basic Type"] == "beverage" and row["Sub-Type 1"] == "energy drink":
        row["Basic Type"] = "energy drink"
        # Note: Subtypes are finicky so we need to actually remove them with the remove_subtypes function
        row = remove_subtypes(row, "energy drink")
        return row

    return row


def process_data(df, **options):
    # Filter missing data and non-food items, handle typos in Category and Group columns
    df = clean_df(df)

    # Create normalized name
    print("Normalizing names...")
    df_normalized = df.progress_apply(clean_name, axis=1)

    # Perform diff on "Normalized Name" column with "Product Name" column from df_loaded
    # Save a diff on the "Product Name" column with the edited output
    print("Creating diff file...")
    df_normalized["Normalized Name"] = df_normalized.progress_apply(
        lambda row: ", ".join(row[NORMALIZED_COLUMNS].dropna().astype(str)),
        axis=1,
    )
    # TODO: do we want more here? Probably should add "Product Type"
    df_diff = df["Product Name"].compare(df_normalized["Normalized Name"])
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

    df_normalized = df_normalized[COLUMNS_ORDER].sort_values(by=TAGS_SORT_ORDER)

    # return processed assets to main
    return df_normalized, misc, df_diff


def main(argv):
    # input
    parser = create_parser()
    # TODO: wait what is this doing?
    options = vars(parser.parse_args(argv))

    # processing
    print("Loading data...")
    df_loaded = load_to_pd(**options)
    df_processed, misc, df_diff = process_data(df_loaded, **options)

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

    # TODO: clean this up and maybe use Chris's save setup
    run_folder_path = Path(CLEAN_FOLDER) / RUN_FOLDER
    run_folder_path.mkdir(parents=True, exist_ok=True)

    diff_file = run_folder_path / "normalized_name_diff.csv"
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
        dict(
            sorted(
                combined_subtype_counts.items(), key=lambda item: item[1], reverse=True
            )
        )
    )

    sorted_counts_dict = {}
    for column in counts_dict.keys():
        if column == "Basic Type":
            sorted_counts_dict[column] = counts_dict[column]
            if "Sub-Types" in counts_dict:
                sorted_counts_dict["Sub-Types"] = counts_dict["Sub-Types"]
        elif column != "Sub-Types":
            sorted_counts_dict[column] = counts_dict[column]

    counts_file = run_folder_path / "value_counts.xlsx"

    # Write the counts to an Excel file
    with pd.ExcelWriter(counts_file) as writer:
        for column, counts in sorted_counts_dict.items():
            df_counts = counts.reset_index()
            df_counts.columns = [column, "Count"]
            df_counts.to_excel(writer, sheet_name=column.replace("/", "_"), index=False)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])

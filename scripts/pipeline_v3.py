import pandas as pd
import os
import argparse

from cgfp.config_tags import (
    CATEGORY_TAGS,
    GROUP_TAGS,
    TOKEN_MAP_DICT,
    SKIP_TOKENS,
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
)

from cgfp.config_pipeline import (
    RAW_FOLDER,
    CLEAN_FOLDER,
    RUN_FOLDER,
    NORMALIZED_COLUMNS,
    GROUP_COLUMNS,
    COLUMNS_ORDER,
)

ALL_FLAVORS = FLAVORS | FRUITS

from cgfp.util import load_to_pd, save_pd_to_csv

from cgfp.config_tags import (
    CATEGORY_TAGS,
    GROUP_TAGS,
    TOKEN_MAP_DICT,
    SKIP_TOKENS,
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
)

from cgfp.config_pipeline import (
    RAW_FOLDER,
    CLEAN_FOLDER,
    RUN_FOLDER,
    NORMALIZED_COLUMNS,
    GROUP_COLUMNS,
    COLUMNS_ORDER,
)

ALL_FLAVORS = FLAVORS | FRUITS


# TODO: maybe this goes somewhere else
def pool_tags(tags_dict):
    for top_level in tags_dict.keys():
        tags_dict[top_level]["All"] = set.union(*tags_dict[top_level].values())
    return tags_dict


GROUP_TAGS = pool_tags(GROUP_TAGS)
CATEGORY_TAGS = pool_tags(CATEGORY_TAGS)


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

    # Remove leading 'PREQUALIFIED: ' string
    df["Product Name"] = df["Product Name"].str.replace(
        "^PREQUALIFIED: ", "", regex=True
    )
    return df


def token_handler(
    token, food_product_group, food_product_category, basic_type, sub_type_1
):
    # Handle edge cases where a token is allowed
    if (
        (token == "blue" and basic_type == "cheese")
        or (token == "instant" and food_product_group == "Beverages")
        or (token == "black" and basic_type in ["tea", "drink"])
        or (token == "gluten free" and sub_type_1 in ["parfait"])
    ):
        return token

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
        return None

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
        return "flavored"

    # Skip flavors and shapes for some basic types
    if basic_type in SKIP_FLAVORS and token in ALL_FLAVORS:
        return None

    if basic_type in SKIP_SHAPE and token in SHAPE_EXTRAS:
        return None

    ### EDGE CASES FOR RENAMING TOKENS ###

    # Map nut tokens to "nut" for some basic types
    if basic_type == "snack" and token in NUTS:
        return "nut"

    # Relabel cheese type as "cheese"
    if (
        food_product_group == "Meals" or basic_type == "snack"
    ) and token in CHEESE_TYPES:
        return "cheese"

    # Map chocolate tokens to "chocolate" for candy
    if basic_type == "candy" and token in CHOCOLATE:
        return "chocolate"

    # "chip" should be mapped to "cut" for pickles...but "chip" is valid for snacks
    if sub_type_1 == "pickle" and token == "chip":
        return "cut"

    # Skip outdated tokens from old name normalization format
    # Do this last since some rules override this
    if token in SKIP_TOKENS:
        return None
    return token


def clean_name(row, group_tags_dict=GROUP_TAGS, category_tags_dict=CATEGORY_TAGS):
    name_list = row["Product Name"].split(",")
    food_product_group = row["Food Product Group"]
    food_product_category = row["Food Product Category"]

    # TODO: Should set this up so that normalized name starts with every column
    # Then we add the tokens to the appropriate column based on membership
    # And maybe we have a list for subtypes, and we parse that at the end of everything
    normalized_name = {col: None for col in NORMALIZED_COLUMNS}
    # TODO: hack to make this work
    normalized_name["Food Product Group"] = food_product_group
    normalized_name["Food Product Category"] = food_product_category
    normalized_name["Product Type"] = row["Product Type"]
    normalized_name["Primary Food Product Category"] = row[
        "Primary Food Product Category"
    ]
    normalized_name["Product Name"] = row["Product Name"]
    misc_col = {"Misc": []}  # make a list so we can append unmatched tokens
    # Initialize sub-type 1 since we need to pass it to token_handler
    sub_type_1 = None
    for i, token in enumerate(name_list):
        token = token.strip()
        token = TOKEN_MAP_DICT.get(token, token)
        # First token is always Basic Type
        if i == 0:
            # TODO: probably a better way to maintain this state...update the name dictionary
            basic_type = token
            normalized_name["Basic Type"] = token
            continue
        # TODO: These are weird edge cases where we are replacing the basic type with a subtype
        # Should this all be done in postprocessing?
        # Handle edge cases for basic type
        if basic_type == "snack" and token in [
            "bar",
        ]:
            basic_type = "bar"
            continue
        if basic_type == "sea salt":
            basic_type = "salt"
            continue

        # TODO: pass row into token handler?
        token = token_handler(
            token, food_product_group, food_product_category, basic_type, sub_type_1
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
                # TODO: This is where the logic for categories is broken
                # Find the category that the token is in and add to normalized_name
                if col in group_tags_dict[food_product_group]:
                    if token in group_tags_dict[food_product_group][col]:
                        normalized_name[col] = token
                        matched = True
                        break
                if col in category_tags_dict.get(food_product_category, {}):
                    if token in category_tags_dict[food_product_category][col]:
                        normalized_name[col] = token
                        matched = True
                        break
            if matched:
                continue
        # First token after basic type is sub-type 1 if it's not from the later tags
        # TODO: set this up so that I'm saving sub-types as a list
        if normalized_name["Sub-Type 1"] is None:
            sub_type_1 = token
            normalized_name["Sub-Type 1"] = sub_type_1
            continue
        elif normalized_name["Sub-Type 2"] is None:
            # elif "Sub-Type 2" not in normalized_name:
            normalized_name["Sub-Type 2"] = token
            continue
        # Aggregate unmatched tokens to add to tag dictionary
        misc_col["Misc"].append(token)
    # TODO: The postprocessing stuff should probably actually go here
    # But...we need the whole row here since we need to check FPG, FPC, etc.
    normalized_name = postprocess_data(normalized_name)

    normalized_name.update(misc_col)

    # TODO: normalized_name vs row is messy here
    return pd.Series(normalized_name)


# TODO: This should go in config
REPLACEMENT_MAP = {
    "fruit": "blend",
    "cheese": "blend",
    "vegetable": "blend",
    "melon": "variety",
}


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


def postprocess_data(row):
    # TODO: Handle sub-type 3 when we add that
    # Count occurrences of each category
    category_counts = {}
    # TODO: subtypes should maybe be in config
    subtypes = ["Sub-Type 1", "Sub-Type 2"]
    for subtype in subtypes:
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
            replacement_value = REPLACEMENT_MAP[category]
            # TODO: Maybe a better way to do this...replace the first category and subsequent ones are None
            replaced = False
            for subtype in subtypes:
                if get_category(row[subtype]) == category:
                    row[subtype] = replacement_value if not replaced else None
                    replaced = True

    ### Handle edge cases for mislabeled data ###
    # "spice" is "Condiments & Snacks"
    if (
        row["Basic Type"] == "spice"
        and row["Food Product Group"] != "Condiments & Snacks"
    ):
        row["Food Product Group"] = "Condiments & Snacks"
        row["Food Product Category"] = "Condiments & Snacks"
        row["Primary Product Category"] = "Condiments & Snacks"

    # "watercress" should be Basic Type, not Sub-Type 1
    if row["Sub-Type 1"] == "watercress" and row["Basic Type"] == "herb":
        row["Basic Type"] = "watercress"
        row["Sub-Type 1"] = None

    return row


def process_data(df, **options):
    # isolates data processing from IO without changing outputs

    # Filter missing data and non-food items, handle typos in Category and Group columns
    df = clean_df(df)

    # Create normalized name
    df_split = df.apply(clean_name, axis=1)
    # TODO: This is here to try to pass the comparison tes
    df_split = df_split.reset_index(drop=True)

    # TODO: Clarify this part...kind of confusing
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
    # misc is now returned and written in `main` outside of data processing path

    TAGS_SORT_ORDER = [
        "Food Product Group",
        "Food Product Category",
        "Basic Type",
        "Sub-Type 1",
        "Sub-Type 2",
    ]

    df_split = df_split[COLUMNS_ORDER].sort_values(by=TAGS_SORT_ORDER)

    # return processed assets to main
    return misc, df_split


def main(argv):
    # input
    parser = create_parser()
    # TODO: wait what is this doing?
    options = vars(parser.parse_args(argv))

    # processing
    df_loaded = load_to_pd(**options)
    misc, df_processed = process_data(df_loaded, **options)

    # output
    # TODO: I...don't get this
    save_pd_to_csv(df_processed, **options)
    save_pd_to_csv(
        misc,
        options.get("clean_folder"),
        options.get("misc_file"),
        output_file="misc.csv",
    )


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])

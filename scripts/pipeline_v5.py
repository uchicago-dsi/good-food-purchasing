import pandas as pd
import os
import argparse
from ordered_set import OrderedSet
from pathlib import Path
from collections import defaultdict

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
    # TODO: put this in config
    # TODO: Do we ever use "Primary Food Product Group?
    df = df[
        [
            "Food Product Category",
            "Primary Food Product Category",
            "Product Type",
            "Product Name",
            "Food Product Group",
        ]
    ].copy()

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
    # TODO: ...should this actually be here??
    # These weird rename rules need to happen before the token handler in order to change the basic type
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


# TODO: Move these mappings to config
basic_type_mapping = {
    "sea salt": ("salt", None),
    "almond": ("nut", "almond"),
    "baba ganoush": ("spread", "baba ganoush"),
    "baklava": ("pastry", "baklava"),
    "banana bread": ("bread", "banana"),
    "barbacoa": ("beef", "barbacoa"),
    "basil": ("herb", "basil"),
    "bell pepper": ("pepper", "bell"),
    "bran": ("wheat bran", None),
    "bratwurst": ("pork", "sausage"),
    "breakfast bar": ("bar", None),
    "brownie": ("dessert", "brownie"),
    "cake": ("dessert", "cake"),
    "cannoli cream": ("filling", "cannoli"),
    "cereal bar": ("bar", None),
    "cheesecake": ("dessert", "cheesecake"),
    "chile": ("pepper", "chile"),
    "chorizo": ("pork", "sausage"),
    "clam juice": ("juice", "clam"),
    "clover sprout": ("sprout", "clover"),
    "club soda": ("soda", "club"),
    "cooking wine": ("wine", "cooking"),
    "cornish hen": ("chicken", "cornish hen"),
    "crème fraiche": ("cream", "fraiche"),
    "cupcake": ("dessert", "cupcake"),
    "danish": ("pastry", "danish"),
    "eggnog": ("drink", "eggnog"),
    "farina": ("cereal", "farina"),
    "frank": ("beef", "frank"),
    "frisee": ("lettuce", "frisee"),
    "fruit basket": ("fruit", "variety"),
    "hog": ("pork", "hog"),
    "honeydew": ("melon", "honeydew"),
    "iced tea": ("tea", "iced"),
    "jalepeno": ("pepper", "jalepeno"),
    "juice slushie": ("juice", "slushie"),
    "ketchup": ("condiment", "ketchup"),
    "marmalade": ("spread", "marmalade"),
    "marshmallow": ("candy", "marshmallow"),
    "miso": ("paste", "miso"),
    "mozzarella": ("cheese", "mozzarella"),
    "nori": ("seaweed", "nori"),
    "peppercorn": ("spice", "peppercorn"),
    "pesto": ("sauce", "pesto"),
    "pig feet": ("pork", "feet"),
    "pita": ("bread", "pita"),
    "potato yam": ("potato", "yam"),
    "prosciutto": ("pork", "prosciutto"),
    "pudding": ("dessert", "pudding"),
    "romaine": ("lettuce", "romaine"),
    "rotini": ("pasta", None),
    "sauerkraut": ("condiment", "sauerkraut"),
    "seasoning tajin": ("seasoning", "tajin"),
    "slush": ("juice", "slushie"),
    "spam": ("pork", "spam"),
    "spring mix": ("lettuce", "spring mix"),
    "squash blossom": ("squash", "blossom"),
    "sunflower": ("seed", "sunflower"),
    "sweet potato": ("potato", "sweet"),
    "tostada": ("shell", "tostada"),
    "trail mix": ("snack", "trail mix"),
    "turnip greens": ("turnip", "greens"),
    "vegetable mix": ("vegetable", "blend"),
    "whipped cream": ("topping", "whipped cream"),
    "cantaloupe": ("melon", "cantaloupe"),
}

for nut in NUTS:
    basic_type_mapping[nut] = ("nut", nut)


def basic_type_handler(row):
    mapping = basic_type_mapping.get(row["Basic Type"], (row["Basic Type"], None))

    basic_type, subtype = mapping
    row["Basic Type"] = basic_type

    if subtype:
        row = add_subtype(row, subtype, first=True)

    return row


def add_subtype(row, token, first=False):
    if first:
        subtypes = OrderedSet([token])
        subtypes.update(row["Sub-Types"])
        row["Sub-Types"] = subtypes
    else:
        row["Sub-Types"].add(token)
    # TODO: maybe you can zip this with the sub-type columns?
    for i, subtype in enumerate(row["Sub-Types"]):
        if i == 0:
            row["Sub-Type 1"] = subtype
        elif i == 1:
            row["Sub-Type 2"] = subtype
        else:
            # Not enough room!
            break
    return row


def clean_name(row, group_tags_dict=GROUP_TAGS, category_tags_dict=CATEGORY_TAGS):
    food_product_group, food_product_category = (
        row["Food Product Group"],
        row["Food Product Category"],
    )
    name_split = row["Product Name"].split(",")
    basic_type = clean_token(name_split[0])
    row["Basic Type"] = basic_type
    row["Sub-Types"] = OrderedSet()
    row["Misc"] = []

    row = basic_type_handler(row)

    for token in name_split[1:]:
        token = clean_token(token)
        token, row = token_handler(token, row)
        if token is None:
            continue  # token_handler returns None for invalid tags so skip
        # TODO: maybe pre-combine the tags?
        # If token is in pre-allowed tags, enter tagging loop
        if token in group_tags_dict.get(food_product_group, {}).get(
            "All", []
        ) or token in category_tags_dict.get(food_product_category, {}).get("All", []):
            # TODO: Create some sort of tags_handler function
            matched = False
            for col in NORMALIZED_COLUMNS:
                # TODO: This is where the logic for categories is broken
                # Should group and category tags be separate or not really?
                # Find the category that the token is in and add to normalized_name
                if col in group_tags_dict[food_product_group]:
                    if token in group_tags_dict[food_product_group][col]:
                        row[col] = token
                        matched = True
                        break
                if col in category_tags_dict.get(food_product_category, {}):
                    if token in category_tags_dict[food_product_category][col]:
                        row[col] = token
                        matched = True
                        break
            if matched:
                continue
        row = add_subtype(row, token)  # Unmatched tokens are subtypes
    row = postprocess_data(row)
    row["Misc"] = list(row["Sub-Types"])[2:] if len(row["Sub-Types"]) > 2 else []
    # row["Misc"].append(subtype)
    return row


# TODO: This should go in config
REPLACEMENT_MAP = {
    "fruit": "fruit",
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


# TODO: Put this in config
SUBTYPE_COLUMNS = ["Sub-Type 1", "Sub-Type 2"]


def postprocess_data(row):
    # TODO: Handle sub-type 3 when we add that
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
                replacement_value = REPLACEMENT_MAP.get(category)

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

    ### Handle edge cases for mislabeled data ###
    # "spice" is always "Condiments & Snacks"
    if (
        row["Basic Type"] == "spice"
        and row["Food Product Group"] != "Condiments & Snacks"
    ):
        row["Food Product Group"] = "Condiments & Snacks"
        row["Food Product Category"] = "Condiments & Snacks"
        row["Primary Product Category"] = "Condiments & Snacks"

    return row


def process_data(df, **options):
    # isolates data processing from IO without changing outputs

    # Filter missing data and non-food items, handle typos in Category and Group columns
    df = clean_df(df)

    # Create normalized name
    df_normalized = df.apply(clean_name, axis=1)
    df_normalized = df_normalized.reset_index(drop=True)

    # TODO: Clarify this part...kind of confusing
    # Save unallocated tags for manual review
    misc = df_normalized[df_normalized["Misc"].apply(lambda x: x != [])][
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
    return misc, df_normalized


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

    # TODO: clean this up and maybe use Chris's save setup
    run_folder_path = Path(CLEAN_FOLDER) / RUN_FOLDER
    run_folder_path.mkdir(parents=True, exist_ok=True)

    # TODO: maybe this should get returned from the pipeline also?
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

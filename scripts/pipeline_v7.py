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
    NORMALIZED_COLUMNS,  # TODO: ...why doesn't this include "Basic Type"?
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

    # TODO: Maybe this goes in config?
    # Handle typos in Primary Food Product Category
    category_typos = {
        "Roots & Tuber": "Roots & Tubers",
    }
    df["Primary Food Product Category"] = df["Primary Food Product Category"].map(
        lambda x: category_typos.get(x, x)
    )

    # Replace "Whole/Minimally Processed" with the value from "Food Product Category"
    df["Primary Food Product Category"] = df.apply(
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


# TODO: config
basic_type_mapping = {
    "sea salt": {"Basic Type": "salt"},
    "almond": {"Basic Type": "nut", "Sub-Types": "almond"},
    "baba ganoush": {"Basic Type": "spread", "Sub-Types": "baba ganoush"},
    "baklava": {"Basic Type": "pastry", "Sub-Types": "baklava"},
    "banana bread": {"Basic Type": "bread", "Sub-Types": "banana"},
    "barbacoa": {"Basic Type": "beef", "Sub-Types": "barbacoa"},
    "basil": {"Basic Type": "herb", "Sub-Types": "basil"},
    "bell pepper": {"Basic Type": "pepper", "Sub-Types": "bell"},
    "bran": {"Basic Type": "wheat bran"},
    "bratwurst": {"Basic Type": "pork", "Sub-Types": "sausage"},
    "breakfast bar": {"Basic Type": "bar"},
    "brownie": {"Basic Type": "dessert", "Sub-Types": "brownie"},
    "cake": {"Basic Type": "dessert", "Sub-Types": "cake"},
    "cannoli cream": {"Basic Type": "filling", "Sub-Types": "cannoli"},
    "cereal bar": {"Basic Type": "bar"},
    "cheesecake": {"Basic Type": "dessert", "Sub-Types": "cheesecake"},
    "chile": {"Basic Type": "pepper", "Sub-Types": "chile"},
    "chorizo": {"Basic Type": "pork", "Sub-Types": "sausage"},
    "clam juice": {"Basic Type": "juice", "Sub-Types": "clam"},
    "clover sprout": {"Basic Type": "sprout", "Sub-Types": "clover"},
    "club soda": {"Basic Type": "soda", "Sub-Types": "club"},
    "cooking wine": {"Basic Type": "wine", "Sub-Types": "cooking"},
    "cornish hen": {"Basic Type": "chicken", "Sub-Types": "cornish hen"},
    "crème fraiche": {"Basic Type": "cream", "Sub-Types": "fraiche"},
    "cupcake": {"Basic Type": "dessert", "Sub-Types": "cupcake"},
    "danish": {"Basic Type": "pastry", "Sub-Types": "danish"},
    "eggnog": {"Basic Type": "drink", "Sub-Types": "eggnog"},
    "farina": {"Basic Type": "cereal", "Sub-Types": "farina"},
    "frank": {"Basic Type": "beef", "Sub-Types": "frank"},
    "frisee": {"Basic Type": "lettuce", "Sub-Types": "frisee"},
    "fruit basket": {"Basic Type": "fruit", "Sub-Types": "variety"},
    "hog": {"Basic Type": "pork", "Sub-Types": "hog"},
    "honeydew": {"Basic Type": "melon", "Sub-Types": "honeydew"},
    "iced tea": {"Basic Type": "tea", "Sub-Types": "iced"},
    "jalepeno": {"Basic Type": "pepper", "Sub-Types": "jalepeno"},
    "juice slushie": {"Basic Type": "juice", "Sub-Types": "slushie"},
    "ketchup": {"Basic Type": "condiment", "Sub-Types": "ketchup"},
    "marmalade": {"Basic Type": "spread", "Sub-Types": "marmalade"},
    "marshmallow": {"Basic Type": "candy", "Sub-Types": "marshmallow"},
    "miso": {"Basic Type": "paste", "Sub-Types": "miso"},
    "mozzarella": {"Basic Type": "cheese", "Sub-Types": "mozzarella"},
    "nori": {"Basic Type": "seaweed", "Sub-Types": "nori"},
    "peppercorn": {"Basic Type": "spice", "Sub-Types": "peppercorn"},
    "pesto": {"Basic Type": "sauce", "Sub-Types": "pesto"},
    "pig feet": {"Basic Type": "pork", "Sub-Types": "feet"},
    "pita": {"Basic Type": "bread", "Sub-Types": "pita"},
    "potato yam": {"Basic Type": "potato", "Sub-Types": "yam"},
    "prosciutto": {"Basic Type": "pork", "Sub-Types": "prosciutto"},
    "pudding": {"Basic Type": "dessert", "Sub-Types": "pudding"},
    "romaine": {"Basic Type": "lettuce", "Sub-Types": "romaine"},
    "rotini": {"Basic Type": "pasta"},
    "sauerkraut": {"Basic Type": "condiment", "Sub-Types": "sauerkraut"},
    "seasoning tajin": {"Basic Type": "seasoning", "Sub-Types": "tajin"},
    "slush": {"Basic Type": "juice", "Sub-Types": "slushie"},
    "spam": {"Basic Type": "pork", "Sub-Types": "spam"},
    "spring mix": {"Basic Type": "lettuce", "Sub-Types": "spring mix"},
    "squash blossom": {"Basic Type": "squash", "Sub-Types": "blossom"},
    "sunflower": {"Basic Type": "seed", "Sub-Types": "sunflower"},
    "sweet potato": {"Basic Type": "potato", "Sub-Types": "sweet"},
    "tostada": {"Basic Type": "shell", "Sub-Types": "tostada"},
    "trail mix": {"Basic Type": "snack", "Sub-Types": "trail mix"},
    "turnip greens": {"Basic Type": "turnip", "Sub-Types": "greens"},
    "vegetable mix": {"Basic Type": "vegetable", "Sub-Types": "blend"},
    "whipped cream": {"Basic Type": "topping", "Sub-Types": "whipped cream"},
    "cantaloupe": {"Basic Type": "melon", "Sub-Types": "cantaloupe"},
    "blend": {"Basic Type": "vegetable", "Sub-Types": "blend"},
    "soy sauce": {"Basic Type": "condiment", "Sub-Types": "soy sauce"},
    "chicken breast": {"Basic Type": "chicken", "Shape": "breast"},
    "chicken tender": {"Basic Type": "chicken", "Shape": "cut"},
    "chocolate": {"Basic Type": "candy", "Shape": "chocolate"},
    "chutney": {"Basic Type": "spread", "Sub-Types": "chutney"},
    "corn nugget": {
        "Basic Type": "appetizer",
        "Sub-Types": "corn",
        "Processing": "battered",
    },
    "gel": {"Basic Type": "topping", "Sub-Types": "icing"},
    "ice cream cone": {"Basic Type": "cone", "Sub-Types": "ice cream"},
    "pan coating": {"Basic Type": "oil", "Sub-Types": "spray"},
    "paprika": {"Basic Type": "spice", "Sub-Types": "paprika"},
    "salad mix": {"Basic Type": "salad", "Sub-Types": "mix"},
    "toast": {"Basic Type": "bread", "Sub-Types": "toast"},
    "turmeric": {"Basic Type": "spice", "Sub-Types": "turmeric"},
    "orange blossom water": {"Basic Type": "water", "Sub-Types": "flavored"},
    "gyro meat": {"Basic Type": "beef", "Sub-Types": ["lamb", "gyro"]},
}

# Add nuts to the mapping
for nut in NUTS:
    basic_type_mapping[nut] = {"Basic Type": "nut", "Sub-Types": nut}


def basic_type_handler(row):
    mapping = basic_type_mapping.get(row["Basic Type"], None)

    if mapping is None:
        return row

    for key, value in mapping.items():
        if key != "Sub-Types":
            row[key] = value

    # TODO: Ok yeah need to update the row and also the subtypes
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
    row = update_subtypes(row)
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
        row = add_subtypes(row, token)  # Unmatched tokens are subtypes

    # split subtypes into columns and store extra tokens in "Misc"
    row = handle_subtypes(row)
    # handle edge cases not captured by other rules
    row = postprocess_data(row)
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


def clear_row(row):
    for col in NORMALIZED_COLUMNS:
        row[col] = None
    row["Sub-Types"] = OrderedSet()
    row["Misc"] = []
    return row


def postprocess_data(row):
    ### Handle edge cases for specific product types ###

    ### Handle edge cases for mislabeled data ###
    # "spice" is always "Condiments & Snacks"
    if (
        row["Basic Type"] == "spice"
        and row["Food Product Group"] != "Condiments & Snacks"
    ):
        row["Food Product Group"] = "Condiments & Snacks"
        row["Food Product Category"] = "Condiments & Snacks"
        row["Primary Product Category"] = "Condiments & Snacks"

    # Note: Subtypes are finicky so we need to actually remove them with the remove_subtypes function
    if row["Basic Type"] == "beverage" and row["Sub-Type 1"] == "energy drink":
        row["Basic Type"] = "energy drink"
        row = remove_subtypes(row, "energy drink")
        return row

    ### Handle mislabeled edge cases based on Product Type ###
    if row["Product Type"] == "HAWAIIAN PUNCH CANS":
        row = clear_row(row)
        row["Basic Type"] = "drink"
        row = add_subtypes(row, "fruit punch")
    elif row["Product Type"] == "Concentrate Chipotle S/O":
        row = clear_row(row)
        row["Basic Type"] = "chipotle"
        row = add_subtypes(row, "chipotle")
    elif row["Product Type"] == "OREGANO AQUARESIN (7.5LB/PL)":
        row = clear_row(row)
        row["Basic Type"] = "spice"
        row = add_subtypes(row, ["oregano", "concentrate"])
    elif row["Product Type"] == "GRAIN, BLEND COUSCOUS TRI-COLOR QUINOA RESEALABLE BAG":
        row = clear_row(row)
        row["Basic Type"] = "quinoa"
        row = add_subtypes(row, ["couscous", "blend"])
    elif row["Product Type"] == "GREEN, MUST CHPD DMSTC IQF FZN":
        row = clear_row(row)
        row["Basic Type"] = "mustard green"
        row["Shape"] = "cut"
        row["Frozen"] = "frozen"
    elif row["Product Type"] == "Turkey Cranberry Snack Sticks":
        row = clear_row(row)
        row["Basic Type"] = "turkey"
        row = add_subtypes(row, "jerky")
    elif row["Product Type"] == "TURKEY STICK SMOKEHOUSE":
        row = clear_row(row)
        row["Basic Type"] = "turkey"
        row = add_subtypes(row, "jerky")
    elif (
        row["Product Type"]
        == "APPETIZER, CHICKEN COCONUT SKEWER .85 OZ COOKED FROZEN KABOB"
    ):
        row = clear_row(row)
        row["Basic Type"] = "chicken"
        row = add_subtypes(row, ["kabob", "coconut"])
        row["Cooked/Cleaned"] = "cooked"
        row["Frozen"] = "frozen"
    elif row["Product Type"] == "APPETIZER, CHICKEN SKEWER .8 OZ PARCOOKED FROZEN":
        row = clear_row(row)
        row["Basic Type"] = "chicken"
        row = add_subtypes(row, "kabob")
        row["Cooked/Cleaned"] = "cooked"
    elif row["Product Type"] == "WG RAMEN MISO NOODLE KIT":
        row = clear_row(row)
        row["Basic Type"] = "meal kit"
        row = add_subtypes(row, ["noodle", "miso"])
        row["WG/WGR"] = "whole grain rich"
    elif row["Product Type"] == "EDAMAME SUCCOTASH BLEND":
        row = clear_row(row)
        row["Basic Type"] = "vegetable"
        row = add_subtypes(row, ["blend", "edamame", "succotash"])
    elif row["Product Type"] == "MIX CAKE BASE RICHCREME":
        row = clear_row(row)
        row["Basic Type"] = "dessert"
        row = add_subtypes(row, ["cake", "mix"])
    elif row["Product Type"] == "QUICK OATS TUBES":
        row = clear_row(row)
        row["Basic Type"] = "oat"
        row = add_subtypes(row, "quick")
    elif row["Product Type"] == "RICE PILAF CHICKEN W/ORZO":
        row = clear_row(row)
        row["Basic Type"] = "entree"
        row = add_subtypes(row, ["chicken", "rice pilaf"])
    elif row["Product Type"] == "GRAIN SPCLTY BULGHUR WHEAT PLF":
        row = clear_row(row)
        row["Basic Type"] = "bulgur"
        row = add_subtypes(row, ["wheat", "pilaf"])
    elif row["Product Type"] == "POLENTA, CAKE VEG FIRE RSTD (6455314)":
        row = clear_row(row)
        row["Basic Type"] = "entree"
        row = add_subtypes(row, ["polenta", "vegetable"])
    elif row["Product Type"] == "ROOT TARO MALANGA COCA":
        row = clear_row(row)
        row["Basic Type"] = "taro"
    elif row["Product Type"] == "SHORTBREAD STRAWBERRY":
        row = clear_row(row)
        row["Basic Type"] = "dessert"
        row = add_subtypes(row, ["shortbread", "strawberry"])
    elif row["Product Type"] == "SCOOBY DOO GRAHAM STIX IW":
        row = clear_row(row)
        row["Basic Type"] = "cracker"
        row = add_subtypes(row, "graham")
        row["WG/WGR"] = "whole grain rich"
        row["Packaging"] = "ss"
    elif row["Product Type"] == "BEAN, GOURMET MADAGASCAR BOURB (4648664)":
        row = clear_row(row)
        row["Basic Type"] = "spice"
        row = add_subtypes(row, "vanilla bean")

    return row


def process_data(df, **options):
    # isolates data processing from IO without changing outputs

    # Filter missing data and non-food items, handle typos in Category and Group columns
    df = clean_df(df)

    # Create normalized name
    df_normalized = df.apply(clean_name, axis=1)

    # Perform diff on "Normalized Name" column with "Product Name" column from df_loaded
    # Save a diff on the "Product Name" column with the edited output
    df_normalized["Normalized Name"] = df_normalized.apply(
        lambda row: ", ".join(
            row[["Basic Type"] + NORMALIZED_COLUMNS].dropna().astype(str)
        ),
        axis=1,
    )
    # TODO: do we want more here?
    df_diff = df["Product Name"].compare(df_normalized["Normalized Name"])
    df_diff = df_diff.sort_values(by="self")

    # Reset index for future sorting
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
    return df_normalized, misc, df_diff


def main(argv):
    # input
    parser = create_parser()
    # TODO: wait what is this doing?
    options = vars(parser.parse_args(argv))

    # processing
    df_loaded = load_to_pd(**options)
    df_processed, misc, df_diff = process_data(df_loaded, **options)

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

    diff_file = run_folder_path / "normalized_name_diff.csv"
    df_diff.to_csv(diff_file, index=False)

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

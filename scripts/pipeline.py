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

if not os.path.exists(CLEAN_FOLDER + RUN_FOLDER):
    os.makedirs(CLEAN_FOLDER + RUN_FOLDER)


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


def clean_name(
    name_list,
    food_product_group,
    food_product_category,
    group_tags_dict,
    category_tags_dict,
):
    # TODO: Should set this up so that normalized name starts with every column
    # Then we add the tokens to the appropriate column based on membership
    normalized_name = {}
    misc_col = {"Misc": []}  # make a list so we can append unmatched tokens
    # Initialize sub-type 1 since we need to pass it to token_handler
    sub_type_1 = None
    for i, token in enumerate(name_list):
        token = token.strip()
        token = TOKEN_MAP_DICT.get(token, token)
        # First token is always Basic Type
        if i == 0:
            basic_type = token
            normalized_name["Basic Type"] = token
            continue
        # Handle edge cases for basic type
        if basic_type == "snack" and token in [
            "bar",
        ]:
            basic_type = "bar"
            continue
        if basic_type == "sea salt":
            basic_type = "salt"
            continue

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
        if "Sub-Type 1" not in normalized_name:
            sub_type_1 = token
            normalized_name["Sub-Type 1"] = sub_type_1
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
    parser = argparse.ArgumentParser(description="Process some files.")

    default_input_file = "CONFIDENTIAL_CGFP bulk data_073123.xlsx"
    default_misc_file = "misc.csv"
    clean_file_prefix = "clean_"

    parser.add_argument(
        "--input_file", default=default_input_file, help="Input file path"
    )
    parser.add_argument(
        "--misc_file", default=default_misc_file, help="Miscellaneous file path"
    )
    parser.add_argument(
        "--clean_file",
        default="",
        help="Clean file path. If not specified, it will be automatically generated based on the input file.",
    )

    args = parser.parse_args()

    # TODO: This doesn't work on the cluster so need to change the configuration for the filepaths
    CLEAN_FILE = clean_file_prefix + args.input_file
    CLEAN_FILE = CLEAN_FILE.replace(" ", "_")

    INPUT_PATH = RAW_FOLDER + args.input_file
    MISC_PATH = CLEAN_FOLDER + RUN_FOLDER + args.misc_file
    CLEAN_PATH = os.path.join(CLEAN_FOLDER, RUN_FOLDER, CLEAN_FILE)
    root, _ = os.path.splitext(CLEAN_PATH)
    CSV_PATH = root + ".csv"

    file_extension = os.path.splitext(INPUT_PATH)[1]
    df = (
        pd.read_excel(INPUT_PATH)
        if file_extension in [".xls", ".xlsx"]
        else pd.read_csv(INPUT_PATH)
    )

    df["Misc"] = None
    df = clean_df(df)

    # Handle any typos or issues with Food Product Category and Primary Food Product Category
    category_typos = {
        "Roots & Tuber": "Roots & Tubers",
    }
    df["Primary Food Product Category"] = df["Primary Food Product Category"].map(
        lambda x: category_typos.get(x, x)
    )

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

    # TODO: All of this should be aggregated in a function that is applied as postprocessing to the dataframe

    # TODO: Handle sub-type 3 when we add that » if more than one sub-type is fruit (or whatever) then replace the string
    # TODO: Maybe want to abstract and functionalize this setup
    # TODO: This should be done with a list for subtypes
    # Replace multiple fruits for juices with "blend"
    juice_blend = (
        (df_split["Basic Type"] == "juice")
        & (df_split["Sub-Type 1"].isin(FRUITS))
        & (df_split["Sub-Type 2"].isin(FRUITS))
    )

    df_split.loc[juice_blend, "Sub Type 1"] = "blend"
    df_split.loc[juice_blend, "Sub Type 2"] = None

    # If anything that is not a fruit has more than one fruit, relabel it as "fruit"
    multiple_fruits = (
        (df_split["Food Product Category"] != "Fruit")
        & (df_split["Sub-Type 1"].isin(FRUITS))
        & (df_split["Sub-Type 2"].isin(FRUITS))
    )
    # TODO: make better logic for this
    # for sparkling water, we want to replace multiple fruits with "flavored"
    # for everything else we want it to be "fruit"
    fruit_water = (df_split["Basic Type"] == "water") & multiple_fruits
    not_fruit_water = (df_split["Basic Type"] != "water") & multiple_fruits

    df_split.loc[fruit_water, "Sub Type 1"] = "flavored"
    df_split.loc[fruit_water, "Sub Type 2"] = None
    df_split.loc[not_fruit_water, "Sub Type 1"] = "fruit"
    df_split.loc[not_fruit_water, "Sub Type 2"] = None

    multiple_cheeses = (
        (df_split["Food Product Category"] == "Cheese")
        & (df_split["Sub-Type 1"].isin(CHEESE_TYPES))
        & (df_split["Sub-Type 2"].isin(CHEESE_TYPES))
    )
    df_split.loc[multiple_fruits, "Sub Type 1"] = "blend"
    df_split.loc[multiple_fruits, "Sub Type 2"] = None

    multiple_veggies = (
        (df_split["Basic Type"] == "vegetable")
        & (df_split["Sub-Type 1"].isin(VEGETABLES))
        & (df_split["Sub-Type 2"].isin(VEGETABLES))
    )
    df_split.loc[multiple_fruits, "Sub Type 1"] = "blend"
    df_split.loc[multiple_fruits, "Sub Type 2"] = None

    multiple_melon = (
        (df_split["Basic Type"] == "melon")
        & (df_split["Sub-Type 1"].isin(MELON_TYPES))
        & (df_split["Sub-Type 2"].isin(MELON_TYPES))
    )
    df_split.loc[multiple_fruits, "Sub Type 1"] = "variety"
    df_split.loc[multiple_fruits, "Sub Type 2"] = None

    # Handle edge cases for mislabeled data
    mask_spice = (df_split["Basic Type"] == "spice") & (
        df_split["Food Product Group"] != "Condiments & Snacks"
    )
    df_split.loc[
        mask_spice,
        ["Food Product Group", "Food Product Category", "Primary Product Category"],
    ] = "Condiments & Snacks"

    # Update 'Basic Type' to 'watercress' and 'Sub-Type 1' to None for entries where 'Sub-Type 1' is 'watercress'
    mask_watercress = (df_split["Sub-Type 1"] == "watercress") & (
        df_split["Basic Type"] == "herb"
    )
    df_split.loc[mask_watercress, "Basic Type"] = "watercress"
    df_split.loc[mask_watercress, "Sub-Type 1"] = None

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
    df_split.to_csv(CSV_PATH, index=False)
    print(f"Pipeline complete! File saved to {CSV_PATH}")

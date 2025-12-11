"""Classify food product attributes with an LLM and export results to CSV."""

import argparse
import csv
import json
import os
import pathlib
import queue
import threading
from functools import reduce
from operator import mul
from typing import Dict, List, Optional, Union

import pandas as pd
import requests
from tqdm import tqdm

# constants

MINIMUM_NUM_SAMPLES = 10
CHATGPT_TIMEOUT = 20  # seconds
CHATGPT_TEMPERATURE = 1.0

with open("/app/p_correct.json") as file:
    P_CORRECT = json.load(file)

with open("/app/p_subtype_correct.json") as file:
    P_SUBTYPE_CORRECT = json.load(file)

ALLOWED = {
    "Food Product Group": [
        "Produce",
        "Condiments & Snacks",
        "Meat",
        "Bread, Grains & Legumes",
        "Meals",
        "Milk & Dairy",
        "Beverages",
        "Non-Food",
        "Seafood",
    ],
    "Food Product Category": [
        "Condiments & Snacks",
        "Vegetables",
        "Meals",
        "Fruit",
        "Grain Products",
        "Beverages",
        "Non-Food",
        "Roots & Tubers",
        "Chicken",
        "Beef",
        "Cheese",
        "Pork",
        "Turkey, Other Poultry",
        "Milk & Dairy",
        "Yogurt",
        "Seafood",
        "Legumes",
        "Milk",
        "Eggs",
        "Tree Nuts & Seeds",
        "Rice",
        "Meat",
        "Butter",
        "Fish (Wild)",
        "Fish (Farm-Raised)",
        "Produce",
    ],
    "Primary Food Product Category": [
        "Condiments & Snacks",
        "Vegetables",
        "Fruit",
        "Grain Products",
        "Beverages",
        "Non-Food",
        "Cheese",
        "Roots & Tubers",
        "Meals",
        "Beef",
        "Chicken",
        "Pork",
        "Turkey, Other Poultry",
        "Milk & Dairy",
        "Seafood",
        "Yogurt",
        "Legumes",
        "Milk",
        "Eggs",
        "Tree Nuts & Seeds",
        "Rice",
        "Butter",
        "Fish (Wild)",
        "Fish (Farm-Raised)",
        "Produce",
        "Meat",
        "Egg",
        "Meats",
    ],
    "Flavor/Cut": [
        "flavored",
        "breast",
        "ham",
        "wing",
        "thigh",
        "steak",
        "loin",
        "rib",
        "mix",
        "tenderloin",
        "leg",
        "brisket",
        "chuck",
        "butt",
        "sirloin",
        "shoulder",
        "short rib",
        "bottom round",
        "belly",
        "shank",
        "oxtail",
        "skirt",
        "tri tip",
        "striploin",
        "knuckle",
        "rack",
        "shortloin",
        "cheek",
        "neck",
        "round",
        "tripe",
        "tongue",
        "teres major",
        "pectoral meat",
        "outside skirt",
        "marrow bone",
        "loin rib",
        "t-bone",
        "cut",
    ],
    "Shape": [
        "cut",
        "patty",
        "ground",
        "concentrate",
        "bacon",
        "hot dog",
        "meatball",
        "thickened",
        "crumble",
        "nugget",
        "jerky",
        "salami",
        "pepperoni",
        "pastrami",
        "bologna",
        "prosciutto",
        "shredded",
        "genoa",
        "liquid",
        "mortadella",
        "capocollo",
        "pancetta",
        "bresaola",
        "sopressata",
        "breast",
        "cotto",
        "guanciale",
        "nostrano",
    ],
    "Skin": [
        "skin on",
        "tail on",
        "shell on",
    ],
    "Seed/Bone": [
        "bone-in",
        "pitted",
    ],
    "Processing": [
        "breaded",
        "in juice",
        "seasoned",
        "dried",
        "in syrup",
        "puree",
        "powder",
        "in water",
        "battered",
        "hard boiled",
        "dehydrated",
        "whipped",
        "grated",
        "corned",
        "in sauce",
        "stuffed",
        "in brine",
        "in oil",
        "evaporated",
        "in puree",
        "in vinegar",
        "in liquid",
        "in gel",
        "marinated",
        "powdered",
        "in vegetable broth",
    ],
    "Cooked/Cleaned": [
        "cooked",
        "smoked",
    ],
    "WG/WGR": [
        "whole grain rich",
    ],
    "Dietary Concern": [
        "nonfat",
        "low sodium",
        "low fat",
        "1%",
        "salted",
        "unsalted",
        "decaffeinated",
        "diet",
        "2%",
        "reduced sodium",
        "fat free",
        "reduced sugar",
        "no sodium",
        "reduced calorie",
        "caffeinated",
    ],
    "Additives": [
        "no additives",
        "unsweetened",
        "additives",
        "sweetened",
    ],
    "Dietary Accommodation": [
        "gluten free",
        "kosher",
        "vegan",
        "vegetarian",
        "lactose free",
        "halal",
        "non-dairy",
    ],
    "Frozen": [
        "frozen",
        "iced",
    ],
    "Packaging": [
        "ss",
        "canned",
        "jarred",
    ],
    "Commodity": [
        "commodity",
    ],
}

SYSTEM_MESSAGE = f"""
Your job is to classify a food product's attributes as a JSON object with the following keys and values:

* "Food Product Group": which must be present and its value must be one of the following: {', '.join(map(json.dumps, ALLOWED['Food Product Group']))}
* "Food Product Category": which must be present and its value must be one of the following: {', '.join(map(json.dumps, ALLOWED['Food Product Category']))}
* "Primary Food Product Category": which must be present and its value must be one of the following: {', '.join(map(json.dumps, ALLOWED['Primary Food Product Category']))}
* "Basic Type": if it is present, it must have a value like "chicken", "beef", "cheese", "juice", "condiment", "pork", "sauce", "potato", "dessert", "pepper", "seasoned", "turkey", "cereal", "chip", "apple", "tomato", "carrot", "dressing", "lettuce", "yogurt", "onion", "bread", "cracker", "herb", "milk", "pasta", "bean", "squash", "bar"
* "Sub-Type": if it is present, it is a list of values like "cheese", "blend", "chicken", "corn", "sausage", "bell", "beef", "potato", "variety", "vegetable", "cake", "mozzarella", "grape", "mayonnaise", "oat", "pie", "sparkling", "syrup", "soy", "chocolate", "cookie", "barbecue", "mustard", "romaine", "graham", "italian", "zucchini", "pepper", "ranch"
* "Flavor/Cut": if present, its value must be one of the following: {', '.join(map(json.dumps, ALLOWED['Flavor/Cut']))}
* "Shape": if present, its value must be one of the following: {', '.join(map(json.dumps, ALLOWED['Shape']))}
* "Skin": if present, its value must be one of the following: {', '.join(map(json.dumps, ALLOWED['Skin']))}
* "Seed/Bone": if present, its value must be one of the following: {', '.join(map(json.dumps, ALLOWED['Seed/Bone']))}
* "Processing": if present, its value must be one of the following: {', '.join(map(json.dumps, ALLOWED['Processing']))}
* "Cooked/Cleaned": if present, its value must be one of the following: {', '.join(map(json.dumps, ALLOWED['Cooked/Cleaned']))}
* "WG/WGR": if present, its value must be one of the following: {', '.join(map(json.dumps, ALLOWED['WG/WGR']))}
* "Dietary Concern": if present, its value must be one of the following: {', '.join(map(json.dumps, ALLOWED['Dietary Concern']))}
* "Additives": if present, its value must be one of the following: {', '.join(map(json.dumps, ALLOWED['Additives']))}
* "Dietary Accommodation": if present, its value must be one of the following: {', '.join(map(json.dumps, ALLOWED['Dietary Accommodation']))}
* "Frozen": if present, its value must be one of the following: {', '.join(map(json.dumps, ALLOWED['Frozen']))}
* "Packaging": if present, its value must be one of the following: {', '.join(map(json.dumps, ALLOWED['Packaging']))}
* "Commodity": if present, its value must be one of the following: {', '.join(map(json.dumps, ALLOWED['Commodity']))}
""".strip()

JSON_SCHEMA = {
    "name": "name_normalization",
    "schema": {
        "type": "object",
        "properties": {
            "Food Product Group": {
                "type": "string",
                "enum": ALLOWED["Food Product Group"],
            },
            "Food Product Category": {
                "type": "string",
                "enum": ALLOWED["Food Product Category"],
            },
            "Primary Food Product Category": {
                "type": "string",
                "enum": ALLOWED["Primary Food Product Category"],
            },
            "Basic Type": {"type": "string"},
            "Sub-Type": {
                "type": "array",
                "items": {"type": "string"},
            },
            "Flavor/Cut": {
                "type": "string",
                "enum": ALLOWED["Flavor/Cut"],
            },
            "Shape": {
                "type": "string",
                "enum": ALLOWED["Shape"],
            },
            "Skin": {
                "type": "string",
                "enum": ALLOWED["Skin"],
            },
            "Seed/Bone": {
                "type": "string",
                "enum": ALLOWED["Seed/Bone"],
            },
            "Processing": {
                "type": "string",
                "enum": ALLOWED["Processing"],
            },
            "Cooked/Cleaned": {
                "type": "string",
                "enum": ALLOWED["Cooked/Cleaned"],
            },
            "WG/WGR": {
                "type": "string",
                "enum": ALLOWED["WG/WGR"],
            },
            "Dietary Concern": {
                "type": "string",
                "enum": ALLOWED["Dietary Concern"],
            },
            "Additives": {
                "type": "string",
                "enum": ALLOWED["Additives"],
            },
            "Dietary Accommodation": {
                "type": "string",
                "enum": ALLOWED["Dietary Accommodation"],
            },
            "Frozen": {
                "type": "string",
                "enum": ALLOWED["Frozen"],
            },
            "Packaging": {
                "type": "string",
                "enum": ALLOWED["Packaging"],
            },
            "Commodity": {
                "type": "string",
                "enum": ALLOWED["Commodity"],
            },
        },
        "required": [
            "Food Product Group",
            "Food Product Category",
            "Primary Food Product Category",
        ],
        "additionalProperties": False,
    },
}

FIELDS = [
    "Index",
    "Product Type",
    "Food Product Group",
    "P(Food Product Group)",
    "Food Product Category",
    "P(Food Product Category)",
    "Primary Food Product Category",
    "P(Primary Food Product Category)",
    "Product Name",
    "P(Product Name)",
    "Basic Type",
    "P(Basic Type)",
    "Sub-Type 1",
    "Sub-Type 2",
    "Sub-Type 3",
    "P(Sub-Types)",
    "Flavor/Cut",
    "P(Flavor/Cut)",
    "Shape",
    "P(Shape)",
    "Skin",
    "P(Skin)",
    "Seed/Bone",
    "P(Seed/Bone)",
    "Processing",
    "P(Processing)",
    "Cooked/Cleaned",
    "P(Cooked/Cleaned)",
    "WG/WGR",
    "P(WG/WGR)",
    "Dietary Concern",
    "P(Dietary Concern)",
    "Additives",
    "P(Additives)",
    "Dietary Accommodation",
    "P(Dietary Accommodation)",
    "Frozen",
    "P(Frozen)",
    "Packaging",
    "P(Packaging)",
    "Commodity",
    "P(Commodity)",
]
FIELD_TO_INDEX = {x: i for i, x in enumerate(FIELDS)}

PRODUCT_NAME_FIELDS = [
    "Basic Type",
    "Sub-Type 1",
    "Sub-Type 2",
    "Sub-Type 3",
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


# functions


def predict_product_name(
    product_type: str, openai_api_key: str
) -> Dict[str, Optional[Union[float, str]]]:
    """Predict normalized attributes for a single product type.

    Args:
        product_type: Free-form product type description from the spreadsheet.
        openai_api_key: API key for the account to charge ChatGPT fees.

    Returns:
        Mapping keyed by `FIELDS` with attribute strings (or empty strings when the
        attribute is missing), unrounded probability floats, or None for missing
        probabilities.
    """
    output: Dict[str, Optional[Union[float, str]]] = {"Product Type": product_type}

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        timeout=CHATGPT_TIMEOUT,
        headers={
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "ft:gpt-4.1-mini-2025-04-14:u-chicago:name-normalization-try3:CgFswafI",
            "temperature": CHATGPT_TEMPERATURE,
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": product_type},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": JSON_SCHEMA,
            },
        },
    )

    result = json.loads(response.json()["choices"][0]["message"]["content"])

    basic_type = None
    for column, probabilities in P_CORRECT.items():
        out = output[column] = result.get(column, "")
        if column == "Basic Type":
            basic_type = out

        if probabilities["numsamples"].get(out, 0) >= MINIMUM_NUM_SAMPLES:
            probability = probabilities["byvalue"].get(out, 0)
        else:
            probability = None
        output[f"P({column})"] = probability

    subtypes = result.get("Sub-Type", [])
    output["Sub-Type 1"] = subtypes[0] if len(subtypes) > 0 else ""
    output["Sub-Type 2"] = subtypes[1] if len(subtypes) > 1 else ""
    output["Sub-Type 3"] = subtypes[2] if len(subtypes) > 2 else ""

    key_suffix = "empty" if len(subtypes) == 0 else "nonempty"
    if (
        P_SUBTYPE_CORRECT[f"numsamples_{key_suffix}"].get(basic_type, 0)
        >= MINIMUM_NUM_SAMPLES
    ):
        probability = P_SUBTYPE_CORRECT[f"byvalue_{key_suffix}"].get(basic_type, 0)
    else:
        probability = None
    output["P(Sub-Types)"] = probability

    product_name_pieces = [output[column] for column in PRODUCT_NAME_FIELDS]
    output["Product Name"] = ", ".join([x for x in product_name_pieces if x != ""])

    probability_factors = [
        output[f"P({column})"]
        for column in PRODUCT_NAME_FIELDS
        if not column.startswith("Sub-Type")
    ] + [output["P(Sub-Types)"]]

    if all(x is not None for x in probability_factors):
        probability = 100 * reduce(mul, [float(x) / 100 for x in probability_factors])
    else:
        probability = None
    output["P(Product Name)"] = probability

    return output


def format_as_output_row(
    output: Dict[str, Optional[Union[float, str]]], output_row: List[str]
) -> None:
    """Format a result dict into a CSV row in-place.

    Args:
        output: Mapping produced by `predict_product_name`.
        output_row: Mutable CSV row to fill; length must equal `FIELDS`.
    """
    for key, value in output.items():
        if not key.startswith("P("):
            output_row[FIELD_TO_INDEX[key]] = value
        else:
            output_row[FIELD_TO_INDEX[key]] = "" if value is None else f"{value:.0f}"


# script


def main() -> None:
    """Parse CLI arguments, orchestrate predictions, and write CSV output."""

    # command line arguments

    parser = argparse.ArgumentParser(
        description="Normalize 'Product Name' in an Excel sheet for Center for Good Food Purchasing food products."
    )
    parser.add_argument(
        "input_excel",
        type=str,
        help="Path to the input Excel spreadsheet file (.xlsx, .xls)",
    )
    parser.add_argument("output_csv", type=str, help="Path to the output CSV file")
    parser.add_argument(
        "--sheet",
        type=str,
        help="Name of the sheet in the Excel file to process (alternative to --sheet-index)",
    )
    parser.add_argument(
        "--sheet-index",
        type=int,
        help="Index of the sheet in the Excel file to process (0-based, alternative to --sheet)",
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=1,
        help="Number of queries to run in parallel",
    )
    args = parser.parse_args()

    if args.sheet is not None and args.sheet_index is not None:
        parser.error("Specify either --sheet or --sheet-index, not both.")
    if args.sheet is not None:
        sheet_kw = args.sheet
    elif args.sheet_index is not None:
        sheet_kw = args.sheet_index
    else:
        sheet_kw = 0

    openai_api_key = input("OpenAI API key: ").strip()

    product_type_sheet = pd.read_excel(args.input_excel, sheet_name=sheet_kw)
    if "Product Type" not in product_type_sheet.columns:
        parser.error("'Product Type' column not found in input spreadsheet.")
    product_type_column = product_type_sheet["Product Type"]

    # parallel-processing

    done_sentinel = object()

    queries: queue.Queue = queue.Queue()
    for index, product_type in enumerate(product_type_column):
        queries.put((index, str(product_type)))

    for _ in range(args.num_parallel):
        queries.put(done_sentinel)

    output_lock = threading.Lock()

    # stream continuously to output file while updating progress bar

    with open(args.output_csv, "w") as output_file:
        output_writer = csv.writer(output_file)
        output_writer.writerow(FIELDS)
        output_file.flush()

        pbar = tqdm(total=len(product_type_column))

        def print_error(err: Exception) -> None:
            print(
                f"{json.dumps(product_type)} failed with {type(err).__name__}: {str(err)}"
            )

        def write_output(output_row: List[str]) -> None:
            with output_lock:
                if not output_file.closed:
                    output_writer.writerow(output_row)
                    output_file.flush()
                pbar.update(1)

        def worker() -> None:
            while True:
                query = queries.get()
                if query is done_sentinel:
                    break  # we're done

                index, product_type = query

                output_row: List[str] = [""] * len(FIELDS)
                output_row[FIELD_TO_INDEX["Index"]] = index
                output_row[FIELD_TO_INDEX["Product Type"]] = product_type

                try:
                    output = predict_product_name(product_type, openai_api_key)
                    format_as_output_row(output, output_row)
                except Exception as err:
                    print_error(err)

                write_output(output_row)

        threads = [threading.Thread(target=worker) for _ in range(args.num_parallel)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        pbar.close()


if __name__ == "__main__":
    main()

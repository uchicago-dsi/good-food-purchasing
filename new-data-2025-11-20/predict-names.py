import argparse
import csv
import json
import os
import pathlib
import queue
import threading
from functools import reduce
from operator import mul

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

MINIMUM_NUM_SAMPLES = 10
OPENAI_API_TIMEOUT = 20  # seconds

if "OPENAI_API_KEY" not in os.environ:
    raise Exception("environment variable `OPENAI_API_KEY` not found")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

DIRECTORY = pathlib.Path(
    "~/Box/dsi-core/11th-hour/good-food-purchasing/nov2025-dataset"
).expanduser()

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
    default=0,
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
sheet_kw = args.sheet if args.sheet is not None else args.sheet_index

product_type_sheet = pd.read_excel(args.input_excel, sheet_name=sheet_kw)
if "Product Type" not in product_type_sheet.columns:
    parser.error("'Product Type' column not found in input spreadsheet.")
product_type_column = product_type_sheet["Product Type"]

with open(DIRECTORY / "p_correct.json") as file:
    p_correct = json.load(file)

with open(DIRECTORY / "p_subtype_correct.json") as file:
    p_subtype_correct = json.load(file)

allowed = {
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

system_message = f"""
Your job is to classify a food product's attributes as a JSON object with the following keys and values:

* "Food Product Group": which must be present and its value must be one of the following: {', '.join(map(json.dumps, allowed['Food Product Group']))}
* "Food Product Category": which must be present and its value must be one of the following: {', '.join(map(json.dumps, allowed['Food Product Category']))}
* "Primary Food Product Category": which must be present and its value must be one of the following: {', '.join(map(json.dumps, allowed['Primary Food Product Category']))}
* "Basic Type": if it is present, it must have a value like "chicken", "beef", "cheese", "juice", "condiment", "pork", "sauce", "potato", "dessert", "pepper", "seasoned", "turkey", "cereal", "chip", "apple", "tomato", "carrot", "dressing", "lettuce", "yogurt", "onion", "bread", "cracker", "herb", "milk", "pasta", "bean", "squash", "bar"
* "Sub-Type": if it is present, it is a list of values like "cheese", "blend", "chicken", "corn", "sausage", "bell", "beef", "potato", "variety", "vegetable", "cake", "mozzarella", "grape", "mayonnaise", "oat", "pie", "sparkling", "syrup", "soy", "chocolate", "cookie", "barbecue", "mustard", "romaine", "graham", "italian", "zucchini", "pepper", "ranch"
* "Flavor/Cut": if present, its value must be one of the following: {', '.join(map(json.dumps, allowed['Flavor/Cut']))}
* "Shape": if present, its value must be one of the following: {', '.join(map(json.dumps, allowed['Shape']))}
* "Skin": if present, its value must be one of the following: {', '.join(map(json.dumps, allowed['Skin']))}
* "Seed/Bone": if present, its value must be one of the following: {', '.join(map(json.dumps, allowed['Seed/Bone']))}
* "Processing": if present, its value must be one of the following: {', '.join(map(json.dumps, allowed['Processing']))}
* "Cooked/Cleaned": if present, its value must be one of the following: {', '.join(map(json.dumps, allowed['Cooked/Cleaned']))}
* "WG/WGR": if present, its value must be one of the following: {', '.join(map(json.dumps, allowed['WG/WGR']))}
* "Dietary Concern": if present, its value must be one of the following: {', '.join(map(json.dumps, allowed['Dietary Concern']))}
* "Additives": if present, its value must be one of the following: {', '.join(map(json.dumps, allowed['Additives']))}
* "Dietary Accommodation": if present, its value must be one of the following: {', '.join(map(json.dumps, allowed['Dietary Accommodation']))}
* "Frozen": if present, its value must be one of the following: {', '.join(map(json.dumps, allowed['Frozen']))}
* "Packaging": if present, its value must be one of the following: {', '.join(map(json.dumps, allowed['Packaging']))}
* "Commodity": if present, its value must be one of the following: {', '.join(map(json.dumps, allowed['Commodity']))}
""".strip()

json_schema = {
    "name": "name_normalization",
    "schema": {
        "type": "object",
        "properties": {
            "Food Product Group": {
                "type": "string",
                "enum": [
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
            },
            "Food Product Category": {
                "type": "string",
                "enum": [
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
            },
            "Primary Food Product Category": {
                "type": "string",
                "enum": [
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
            },
            "Basic Type": {"type": "string"},
            "Sub-Type": {
                "type": "array",
                "items": {"type": "string"},
            },
            "Flavor/Cut": {
                "type": "string",
                "enum": [
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
            },
            "Shape": {
                "type": "string",
                "enum": [
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
            },
            "Skin": {
                "type": "string",
                "enum": ["skin on", "tail on", "shell on"],
            },
            "Seed/Bone": {
                "type": "string",
                "enum": ["bone-in", "pitted"],
            },
            "Processing": {
                "type": "string",
                "enum": [
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
            },
            "Cooked/Cleaned": {
                "type": "string",
                "enum": ["cooked", "smoked"],
            },
            "WG/WGR": {
                "type": "string",
                "enum": ["whole grain rich"],
            },
            "Dietary Concern": {
                "type": "string",
                "enum": [
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
            },
            "Additives": {
                "type": "string",
                "enum": ["no additives", "unsweetened", "additives", "sweetened"],
            },
            "Dietary Accommodation": {
                "type": "string",
                "enum": [
                    "gluten free",
                    "kosher",
                    "vegan",
                    "vegetarian",
                    "lactose free",
                    "halal",
                    "non-dairy",
                ],
            },
            "Frozen": {
                "type": "string",
                "enum": ["frozen", "iced"],
            },
            "Packaging": {
                "type": "string",
                "enum": ["ss", "canned", "jarred"],
            },
            "Commodity": {
                "type": "string",
                "enum": ["commodity"],
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

fields = [
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
field_to_index = {x: i for i, x in enumerate(fields)}

product_name_fields = [
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

done_sentinel = object()

queries = queue.Queue()
for product_type in product_type_column:
    queries.put(product_type)

for _ in range(args.num_parallel):
    queries.put(done_sentinel)

output_lock = threading.Lock()

with open(args.output_csv, "w") as output_file:
    output_writer = csv.writer(output_file)
    output_writer.writerow(fields)
    output_file.flush()

    pbar = tqdm(total=len(product_type_column))

    def print_error(err):
        print(
            f"{json.dumps(product_type)} failed with {type(err).__name__}: {str(err)}"
        )

    def write_output(output_row):
        with output_lock:
            output_writer.writerow(output_row)
            output_file.flush()
            pbar.update(1)

    def worker():
        while True:
            product_type = queries.get()
            if product_type is done_sentinel:
                break

            output_row = [""] * len(fields)
            output_row[field_to_index["Product Type"]] = product_type

            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    timeout=OPENAI_API_TIMEOUT,
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "ft:gpt-4.1-mini-2025-04-14:u-chicago:name-normalization-try3:CgFswafI",
                        "messages": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": product_type},
                        ],
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": json_schema,
                        },
                    },
                )
            except Exception as err:
                print_error(err)
                write_output(output_row)
                continue

            try:
                response_json = response.json()
            except Exception as err:
                print_error(err)
                write_output(output_row)
                continue

            try:
                output = json.loads(response_json["choices"][0]["message"]["content"])
            except Exception as err:
                print_error(err)
                write_output(output_row)
                continue

            basic_type = None
            for column, probabilities in p_correct.items():
                out = output_row[field_to_index[column]] = output.get(column, "")
                if column == "Basic Type":
                    basic_type = out

                if probabilities["numsamples"].get(out, 0) >= MINIMUM_NUM_SAMPLES:
                    probability = f"{probabilities['byvalue'].get(out, 0):.0f}"
                else:
                    probability = "???"
                output_row[field_to_index[f"P({column})"]] = probability

            subtypes = output.get("Sub-Type", [])
            if len(subtypes) > 0:
                output_row[field_to_index["Sub-Type 1"]] = subtypes[0]
            if len(subtypes) > 1:
                output_row[field_to_index["Sub-Type 2"]] = subtypes[1]
            if len(subtypes) > 2:
                output_row[field_to_index["Sub-Type 3"]] = subtypes[2]

            key_suffix = "empty" if len(subtypes) == 0 else "nonempty"
            if (
                p_subtype_correct[f"numsamples_{key_suffix}"].get(basic_type, 0)
                >= MINIMUM_NUM_SAMPLES
            ):
                probability = (
                    f"{p_subtype_correct[f'byvalue_{key_suffix}'].get(basic_type, 0):.0f}"
                )
            else:
                probability = "???"
            output_row[field_to_index["P(Sub-Types)"]] = probability

            product_name_pieces = [
                output_row[field_to_index[column]] for column in product_name_fields
            ]
            output_row[field_to_index["Product Name"]] = ", ".join(
                [x for x in product_name_pieces if x != ""]
            )

            probability_factors = [
                output_row[field_to_index[f"P({column})"]]
                for column in product_name_fields
                if not column.startswith("Sub-Type")
            ] + [output_row[field_to_index["P(Sub-Types)"]]]

            if all(x != "???" for x in probability_factors):
                probability = f"{100 * reduce(mul, [float(x) / 100 for x in probability_factors]):.0f}"
            else:
                probability = "???"
            output_row[field_to_index["P(Product Name)"]] = probability

            write_output(output_row)

    threads = [threading.Thread(target=worker) for _ in range(args.num_parallel)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    pbar.close()

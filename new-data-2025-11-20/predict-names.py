"""Classify food product attributes with an LLM and export results to CSV."""

import argparse
import csv
import json
import re
import time
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import requests

# constants

CHATGPT_TEMPERATURE = 1.0

FIELDS = [
    "Index",
    "Product Type",
    "Food Product Group",
    "Food Product Category",
    "Primary Food Product Category",
    "Product Name",
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
FIELD_TO_INDEX = {x: i for i, x in enumerate(FIELDS)}

CATEGORY_TO_GROUP = {
    "Beverages": "Beverages",
    "Grain Products": "Bread, Grains & Legumes",
    "Legumes": "Bread, Grains & Legumes",
    "Rice": "Bread, Grains & Legumes",
    "Tree Nuts & Seeds": "Bread, Grains & Legumes",
    "Condiments & Snacks": "Condiments & Snacks",
    "Meals": "Meals",
    "Beef": "Meat",
    "Chicken": "Meat",
    "Eggs": "Meat",
    "Meat": "Meat",
    "Pork": "Meat",
    "Turkey, Other Poultry": "Meat",
    "Butter": "Milk & Dairy",
    "Cheese": "Milk & Dairy",
    "Milk": "Milk & Dairy",
    "Milk & Dairy": "Milk & Dairy",
    "Yogurt": "Milk & Dairy",
    "Non-Food": "Non-Food",
    "Fruit": "Produce",
    "Produce": "Produce",
    "Roots & Tubers": "Produce",
    "Vegetables": "Produce",
    "Fish (Farm-Raised)": "Seafood",
    "Fish (Wild)": "Seafood",
    "Seafood": "Seafood",
}

with open("/app/category-instructions.md") as file:
    category_instructions = file.read()

with open("/app/category-schema.json") as file:
    category_schema = json.load(file)

with open("/app/tag-instructions.md") as file:
    tag_instructions = file.read()

with open("/app/tag-schema.json") as file:
    tag_schema = json.load(file)

# functions


def chatgpt_for_categories(
    food_products: Sequence[Dict[str, Any]],
    openai_api_key: str,
    chatgpt_timeout: int,
) -> List[Dict[str, Any]]:
    """Fetch product categories for the provided food product inputs."""
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        timeout=chatgpt_timeout,
        headers={
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "ft:gpt-4.1-mini-2025-04-14:u-chicago:cgfp-category-try2:CphJxpT2",
            "temperature": CHATGPT_TEMPERATURE,
            "messages": [
                {
                    "role": "system",
                    "content": f"""
Your job is to read a set of food product names and categorize them. The input
is formatted as

```json
{{"food_products": [...]}}
```

where each object in `...` contains a food product description in its `"input"`.
You need to produce a similar JSON object in which each of the output `"food_products"`
corresponds to one of the inputs, repeating the `"input"` value exactly and adding
category attributes as described below.

{category_instructions}
""".strip(),
                },
                {
                    "role": "user",
                    "content": json.dumps({"food_products": food_products}),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "cgfp-categorization",
                    "schema": category_schema,
                },
            },
        },
    )

    if response.status_code != 200:
        raise Exception(
            f"ChatGPT raised error status code {response.status_code}:\n\n{response.text}"
        )

    data = response.json()
    if len(data.get("choices", [])) == 0:
        raise Exception(f'ChatGPT didn\'t return any "choices":\n\n{response.text}')

    data2 = json.loads(data["choices"][0].get("message", {}).get("content", "null"))
    if not isinstance(data2.get("food_products"), list):
        raise Exception(
            f"ChatGPT returned data with the wrong format:\n\n{json.dumps(data2, indent=2)}"
        )

    return data2["food_products"]


def chatgpt_for_tags(
    food_products: Sequence[Dict[str, Any]],
    openai_api_key: str,
    chatgpt_timeout: int,
) -> List[Dict[str, Any]]:
    """Fetch product tags for the provided food product inputs."""
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        timeout=chatgpt_timeout,
        headers={
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "ft:gpt-4.1-mini-2025-04-14:u-chicago:cgfp-tag-try2:CphN8ams",
            "temperature": CHATGPT_TEMPERATURE,
            "messages": [
                {
                    "role": "system",
                    "content": f"""
Your job is to read a set of food product names and assign tags to them. The input
is formatted as

```json
{{"food_products": [...]}}
```

where each object in `...` contains a food product description in its `"input"`.
You need to produce a similar JSON object in which each of the output `"food_products"`
corresponds to one of the inputs, repeating the `"input"` value exactly and adding
tag attributes as described below.

{tag_instructions}
""".strip(),
                },
                {
                    "role": "user",
                    "content": json.dumps({"food_products": food_products}),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "cgfp-categorization",
                    "schema": tag_schema,
                },
            },
        },
    )

    if response.status_code != 200:
        raise Exception(
            f"ChatGPT raised error status code {response.status_code}:\n\n{response.text}"
        )

    data = response.json()
    if len(data.get("choices", [])) == 0:
        raise Exception(f'ChatGPT didn\'t return any "choices":\n\n{response.text}')

    data2 = json.loads(data["choices"][0].get("message", {}).get("content", "null"))
    if not isinstance(data2.get("food_products"), list):
        raise Exception(
            f"ChatGPT returned data with the wrong format:\n\n{json.dumps(data2, indent=2)}"
        )

    return data2["food_products"]


def categories_and_tags_to_fields(
    index: int,
    categories: Dict[str, Any],
    tags: Dict[str, Any],
) -> List[str]:
    """Convert category and tag payloads into a CSV row aligned to FIELDS."""
    output = [""] * len(FIELDS)
    output[FIELD_TO_INDEX["Index"]] = index
    output[FIELD_TO_INDEX["Product Type"]] = categories.get("input", "???")

    category = categories.get("category", "???")
    subcategory = categories.get("subcategory", "")
    if subcategory == "":
        subcategory = category
    group = CATEGORY_TO_GROUP.get(category, "???")
    output[FIELD_TO_INDEX["Food Product Group"]] = group
    output[FIELD_TO_INDEX["Food Product Category"]] = category
    output[FIELD_TO_INDEX["Primary Food Product Category"]] = subcategory

    sub_types = tags.get("sub_types", [])
    if not isinstance(sub_types, list):
        sub_types = [sub_types]
    output[FIELD_TO_INDEX["Basic Type"]] = tags.get("basic_type", "???")
    output[FIELD_TO_INDEX["Sub-Type 1"]] = sub_types[0] if len(sub_types) > 0 else ""
    output[FIELD_TO_INDEX["Sub-Type 2"]] = sub_types[1] if len(sub_types) > 1 else ""
    output[FIELD_TO_INDEX["Sub-Type 3"]] = sub_types[2] if len(sub_types) > 2 else ""

    output[FIELD_TO_INDEX["Flavor/Cut"]] = (
        ("flavored" if tags.get("flavored", False) else "")
        + ("/" if "flavored" in tags and "meat_cut" in tags else "")
        + tags.get("meat_cut", "")
    )

    output[FIELD_TO_INDEX["Shape"]] = tags.get("shape", "")
    output[FIELD_TO_INDEX["Skin"]] = tags.get("meat_skin", "")

    output[FIELD_TO_INDEX["Seed/Bone"]] = (
        ("pitted" if tags.get("seed_pitted", False) else "")
        + ("/" if "seed_pitted" in tags and "meat_bone" in tags else "")
        + ("bone-in" if tags.get("meat_bone", False) else "")
    )

    output[FIELD_TO_INDEX["Processing"]] = tags.get("processing", "")
    output[FIELD_TO_INDEX["Cooked/Cleaned"]] = tags.get("cooked", "")
    output[FIELD_TO_INDEX["WG/WGR"]] = (
        "whole grain rich" if tags.get("whole_grain", False) else ""
    )

    output[FIELD_TO_INDEX["Dietary Concern"]] = "/".join(
        [
            x
            for x in [
                tags.get("fat_content", ""),
                tags.get("sodium_level", ""),
                tags.get("caffeine", ""),
                tags.get("diet", ""),
                "reduced sugar" if tags.get("reduced_sugar", False) else "",
            ]
            if x != ""
        ]
    )

    output[FIELD_TO_INDEX["Additives"]] = (
        tags.get("sweetened", "")
        + ("/" if "sweetened" in tags and "additives" in tags else "")
        + tags.get("additives", "")
    )

    output[FIELD_TO_INDEX["Dietary Accommodation"]] = tags.get(
        "dietary_accommodation", ""
    )
    output[FIELD_TO_INDEX["Frozen"]] = tags.get("frozen", "")
    output[FIELD_TO_INDEX["Packaging"]] = tags.get("packaging", "")
    output[FIELD_TO_INDEX["Commodity"]] = (
        "commodity" if tags.get("commodity", False) else ""
    )

    output[FIELD_TO_INDEX["Product Name"]] = ", ".join(
        [
            x
            for x in [
                output[FIELD_TO_INDEX["Basic Type"]],
                output[FIELD_TO_INDEX["Sub-Type 1"]],
                output[FIELD_TO_INDEX["Sub-Type 2"]],
                output[FIELD_TO_INDEX["Sub-Type 3"]],
                output[FIELD_TO_INDEX["Flavor/Cut"]],
                output[FIELD_TO_INDEX["Shape"]],
                output[FIELD_TO_INDEX["Skin"]],
                output[FIELD_TO_INDEX["Seed/Bone"]],
                output[FIELD_TO_INDEX["Processing"]],
                output[FIELD_TO_INDEX["Cooked/Cleaned"]],
                output[FIELD_TO_INDEX["WG/WGR"]],
                output[FIELD_TO_INDEX["Dietary Concern"]],
                output[FIELD_TO_INDEX["Additives"]],
                output[FIELD_TO_INDEX["Dietary Accommodation"]],
                output[FIELD_TO_INDEX["Frozen"]],
                output[FIELD_TO_INDEX["Packaging"]],
                output[FIELD_TO_INDEX["Commodity"]],
            ]
            if x != ""
        ]
    )

    return output


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
        "--chatgpt-timeout",
        type=int,
        default=300,
        help="Number of seconds to wait for a ChatGPT response before giving up",
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

    print(f"Reading Excel file {args.input_excel}", flush=True)

    product_type_sheet = pd.read_excel(args.input_excel, sheet_name=sheet_kw)
    if "Product Type" not in product_type_sheet.columns:
        parser.error("'Product Type' column not found in input spreadsheet.")
    product_type_column = product_type_sheet["Product Type"]

    rng = np.random.default_rng()
    all_index = rng.permutation(np.arange(len(product_type_column)))
    food_products = [
        {"input": re.sub(r"\s+", " ", x.upper())}
        for x in product_type_column.iloc[all_index]
    ]

    print(
        f"Sending {len(food_products)} food products to ChatGPT for categorizing",
        flush=True,
    )
    start_timer = time.time()
    all_categories = chatgpt_for_categories(
        food_products, openai_api_key, args.chatgpt_timeout
    )
    sec = int(round(time.time() - start_timer))
    print(
        f"Got {len(all_categories)} categorized food products back in {sec} seconds",
        flush=True,
    )

    print(
        f"Sending {len(food_products)} food products to ChatGPT for tagging", flush=True
    )
    start_timer = time.time()
    all_tags = chatgpt_for_tags(food_products, openai_api_key, args.chatgpt_timeout)
    sec = int(round(time.time() - start_timer))
    print(f"Got {len(all_tags)} tagged food products back in {sec} seconds", flush=True)

    print(f"Writing output CSV file: {args.output_csv}", flush=True)
    with open(args.output_csv, "w") as output_file:
        output_writer = csv.writer(output_file)
        output_writer.writerow(FIELDS)

        for index, categories, tags in zip(all_index, all_categories, all_tags):
            output_writer.writerow(
                categories_and_tags_to_fields(index, categories, tags)
            )

    print("Done!", flush=True)


if __name__ == "__main__":
    main()

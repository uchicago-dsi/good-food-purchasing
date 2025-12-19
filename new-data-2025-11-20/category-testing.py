import json
import os
import pathlib
import threading
import queue
import sys

import numpy as np
import requests
from tqdm import tqdm

NUM_THREADS = 10

DIRECTORY = pathlib.Path("~/Box/dsi-core/11th-hour/good-food-purchasing/nov2025-dataset/").expanduser()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TIMEOUT = 20  # seconds
TEMPERATURE = 1.0  # randomness from 0 to 2

model, = sys.argv[1:]

with open("category-instructions.md") as file:
    category_instructions = file.read()

with open("category-schema.json") as file:
    category_schema = json.load(file)

category_testing = []
with open(DIRECTORY / "fine-tuning-2" / "category-testing-try1.jsonl") as file:
    for line in file:
        category_testing.append(json.loads(line))

tasks = queue.Queue()
for message_pair in category_testing:
    tasks.put(message_pair["messages"])

for _ in range(NUM_THREADS):
    tasks.put(None)

output_file = open(f"category-{model}.jsonl", "w")
output_lock = threading.Lock()

pbar = tqdm(total=len(category_testing))

def worker():
    while True:
        task = tasks.get()
        if task is None:
            break

        user_message, expected_output = task

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                timeout=TIMEOUT,
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "temperature": TEMPERATURE,
                    "messages": [
                        {"role": "system", "content": f"""
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
""".strip()},
                        user_message,
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
        except Exception as err:
            print(f"{type(err).__name__}: {str(err)}")
            continue

        try:
            actual_output = json.loads(response.json()["choices"][0]["message"]["content"])
        except Exception:
            print(response.text)
            continue

        output_line = json.dumps(
            {"actual": actual_output, "expected": json.loads(expected_output["content"])}
        ) + "\n"
        with output_lock:
            output_file.write(output_line)
            output_file.flush()
            pbar.update(1)

threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

pbar.close()
output_file.close()

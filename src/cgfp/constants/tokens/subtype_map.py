"""Defines sub-type mappings for specific tokens in CGFP tagging system"""

SUBTYPE_MAP = {
    "2% lactose free": (None, {"Dietary Accommodation": "lactose free", "Dietary Concern": "2%"}),
    "apple juice": ("apple", {"Basic Type": "juice"}),
    "cheez-it": ("cheese", {"Basic Type": "cracker"}),
    "french toast bread": (None, {"Basic Type": "french toast"}),
    "fruit bar": ("fruit", {"Basic Type": "popsicle"}),
    "gherkin": ("pickle", {"Basic Type": "condiment"}),
    "gravy master": ("browning", {"Basic Type": "sauce"}),
    "whole grain rich ss": (None, {"WG/WGR": "whole grain rich", "Packaging": "ss"}),
    "ham diced": ("ham", {"Basic Type": "pork", "Shape": "cut"}),
    "hanger steak": ("steak", {"Basic Type": "beef", "Shape": "cut"}),
    "low fat ss": (None, {"Dietary Concern": "low fat", "Packaging": "ss"}),
    "mucho queso": ("cheese", {"Basic Type": "sauce"}),
    "nutter butter": ("cookie", {"Basic Type": "snack"}),
    "vegan mayonnaise": ("mayonnaise", {"Dietary Accommodation": "vegan"}),
}

MULTIPLE_SUBTYPES_MAP = {
    "fried onion": {
        "basic_type": "topping",
        "subtypes": ["onion", "fried"],
        "first_subtype": True,  # You can indicate whether 'first' should be applied
    },
    "long grain and wild": {"subtypes": ["long grain", "wild"]},
    "pea & carrot": {"subtypes": ["pea", "carrot"]},
    "pea and carrot": {"subtypes": ["pea", "carrot"]},
    "fruit and nut": {"subtypes": ["fruit", "nut"]},
    "ham and cheese": {"subtypes": ["ham", "cheese"]},
    "mozzarella provolone": {"subtypes": ["mozzarella", "provolone"]},
    "crispix": {"subtypes": ["rice", "corn"]},
    "cap'n crunch": {"subtypes": ["corn", "oat"]},
    "bean hummus": {"subtypes": ["bean", "hummus"]},
}

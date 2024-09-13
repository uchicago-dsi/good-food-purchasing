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
    "dried banana": ("banana", {"Processing": "dried"}),
    "jerk chicken": ("chicken", {"Processing": "seasoned"}),
    "chicken strips": ("chicken", {"Shape": "cut"}),
    "scrambled eggs": ("egg", {"Cooked/Cleaned": "cooked"}),
    "lemon pepper fish": ("fish", {"Processing": "seasoned"}),
    "blueberry cream cheese": ("cream cheese", {"Basic Type": "cheese", "Flavor/Cut": "flavored"}),
    "veggie patty": ("vegetable", {"Shape": "patty"}),
    "marrow bone pipe": (None, {"Flavor/Cut": "marrow bone", "Shape": "cut"}),
    "green commodity": ("bell", {"Commodity": "commodity"}),
    "gluten free ss": (None, {"Dietary Accommodation": "gluten free", "Packaging": "ss"}),
}

MULTIPLE_SUBTYPES_MAP = {
    "fried onion": {
        "basic_type": "topping",
        "subtypes": ["onion", "fried"],
        "first_subtype": True,
    },
    "long grain and wild": {"subtypes": ["long grain", "wild"]},
    "pea & carrot": {"subtypes": ["pea", "carrot"]},
    "pea and carrot": {"subtypes": ["pea", "carrot"]},
    "fruit and nut": {"subtypes": ["fruit", "nut"]},
    "ham and cheese": {"subtypes": ["ham", "cheese"]},
    "mozzarella provolone": {"subtypes": ["mozzarella", "provolone"]},
    "bean hummus": {"subtypes": ["bean", "hummus"]},
    "vegetarian chili": {"subtypes": ["chili", "vegetable"], "first_subtype": True},
    "spicy crab roll": {"subtypes": ["crab", "roll"]},
    "cheetos": {"subtypes": ["corn", "cheese"]},
    "barley. instant": {"subtypes": ["barley", "instant"]},
}
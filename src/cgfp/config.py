# More complicated conditional rules
# TODO: Find rules for converting flavors to "flavored"
# TODO: some of the cheese is incorrect in the dataset » or wait, this is all "milk & dairy"
# TODO: cheese types? "manchego", "mozzarella", "blue", etc.
# TODO: "blue" is a problem for colors and cheese » what about "white" for vegetables? "onion" but maybe we want this for other ones
# also white corn, etc.
# TODO: "whole weat" for pasta? ravioli?

TOKEN_MAP_DICT = {
    ## TYPOS ##
    "whole grain rich  rich": "whole grain rich",
    "orchiette": "orecchiette",
    "campanelli": "campanelle",
    "unsweeted": "unsweetened",
    "peeled & deveined": "peeled and deveined",
    "mangu": "mango",
    ## INCONSISTENCIES ##
    "skin-on": "skin on",
    "carrots": "carrot",
    "gluten-free": "gluten free",
    "tail-on": "tail on",
    "tail-off": "tail off",
    "sugar-free": "sugar free",
    ## CUT ##
    "sliced": "cut",
    "diced": "cut",
    "chopped": "cut",
    "wedge": "cut",
    "segment": "cut",
    ## WHOLE GRAIN ##
    "whole wheat": "whole grain rich",
    ## COOKED ##
    "baked": "cooked",
    "fried": "cooked",
    "roasted": "cooked",
    "parboiled": "cooked",
    "parcooked": "cooked",
    "broiled": "cooked",
    "parbaked": "cooked",
    "broiled": "cooked",
    "parfried": "cooked",
    ## FLAVORED ##
    ## PACKAGING ##
    "bag": "ss",
}

SKIP_TOKENS = {
    ## PRODUCE ##
    "baby",
    ## BRAND NAMES ##
    "cheerios",
    "coke",
    "pikes place",
    ## COLORS ##
    "red",
    "yellow",
    "green",
    "gold",
    "white",
    # "blue", # TODO: we want to keep this for cheese so create a rule
    ## BONELESS ##
    "boneless",
    ## PASTA TYPES ##
    "elbow",
    "rigatoni",
    "angel hair",
    "acini de pepe",
    "bow tie",
    "campanelle",
    "capelli",
    "cavatappi",
    "ditalini",
    "farfalle",
    "fetuccine",
    "fusilli",
    "gnocchi",
    "lasagna",
    "macaroni",
    "manicotti",
    "orzo",
    "pappardelle",
    "penne",
    "penne rigate",
    "pennette",
    "rigate",
    "rigatoni",
    "rotini",
    "shell",
    "spaetzle",
    "spaghetti",
    "stuffed shell",
    "tortellini",
    "vermicelli",
    "ziti",
    ## PACKAGING ##
    "aerosol",
}

# FOOD PRODUCT GROUPS #
"""
Produce
Milk & Dairy
Meat
Seafood
Beverages
Meals
Condiments & Snacks
"""

GROUP_TAGS = {
    "Beverages": {
        "Flavor/Cut": {"flavored"},
        "Shape": {"thickened", "concentrate"},
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": set(),
        "Cooked/Cleaned": set(),
        "WG/WGR": set(),
        "Dietary Concern": {"low sodium", "reduced sugar"},
        "Additives": {"no additives"},
        "Dietary Accommodation": {"gluten free", "non-dariy", "vegan"},
        "Frozen": {"frozen/iced"},
        "Packaging": {"ss"},
        "Commodity": {"commodity"},
    },
    "Bread, Grains & Legumes": {
        "Flavor/Cut": {"flavored"},
        "Shape": set(),
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": {"bleached", "dried", "halved"},
        "Cooked/Cleaned": {"oil roasted", "parbaked", "roasted"},
        "WG/WGR": {"whole grain rich"},
        "Dietary Concern": {"salted", "unsalted"},
        "Additives": {"no additives"},
        "Dietary Accommodation": {"gluten free"},
        "Frozen": {"frozen"},
        "Packaging": {"ss"},
        "Commodity": {"commodity"},
    },
    "Condiments & Snacks": {
        "Flavor/Cut": {"flavored"},
        "Shape": set(),
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": {"diced", "dried", "in juice", "sliced"},
        "Cooked/Cleaned": {"baked", "cooked", "fried", "kettle cooked"},
        "WG/WGR": {"whole grain rich"},
        "Dietary Concern": {"low sodium", "no sugar added", "reduced fat"},
        "Additives": {"no additives", "additives"},
        "Dietary Accommodation": {"gluten free", "kosher", "vegan", "vegetarian"},
        "Frozen": {"frozen"},
        "Packaging": {"ss", "canned"},
        "Commodity": {"commodity"},
    },
    "Meals": {
        "Flavor/Cut": set(),
        "Shape": set(),
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": {"breaded", "condensed", "dehydrated"},
        "Cooked/Cleaned": {"fried", "roasted"},
        "WG/WGR": {"whole grain rich"},
        "Dietary Concern": {"reduced fat", "reduced sodium"},
        "Additives": {"no additives"},
        "Dietary Accommodation": {"halal", "kosher", "vegan", "vegetarian"},
        "Frozen": set(),
        "Packaging": {"canned", "ss"},
        "Commodity": {"commodity"},
    },
    "Meat": {
        "Flavor/Cut": {
            "brisket",
            "breast",
            "loin",
            "shank",
            "skirt",
            "steak",
            "short rib",
            "t-bone",
            "thigh",
            "whole",
            "wing",
        },
        "Shape": {
            "bacon",
            "chunk",
            "diced",
            "fillet",
            "ground",
            "ham",
            "hot dog",
            "jerky",
            "link",
            "meatball",
            "nugget",
            "pepperoni",
            "patty",
            "popcorn",
            "portion",
            "pulled",
            "salami",
            "strip",
            "taco meat",
            "tender",
        },
        "Skin": set(),
        "Seed/Bone": {"bone-in"},
        "Processing": {
            "au jus",
            "battered",
            "breaded",
            "corned",
            "seasoned",
            "shredded",
        },
        "Cooked/Cleaned": {"charbroiled", "cooked", "fried", "roasted"},
        "WG/WGR": set(),
        "Dietary Concern": {"low sodium"},
        "Additives": {"no additives"},
        "Dietary Accommodation": {"gluten free", "halal", "kosher"},
        "Frozen": {"frozen"},
        "Packaging": set(),
        "Commodity": {"commodity"},
    },
    "Milk & Dairy": {
        "Flavor/Cut": set(),
        "Shape": set(),
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": {"condensed", "evaporated", "powdered"},
        "Cooked/Cleaned": set(),
        "WG/WGR": set(),
        "Dietary Concern": set(),
        "Additives": {"no additives"},
        "Dietary Accommodation": {"lactose free"},
        "Frozen": {"frozen"},
        "Packaging": {"ss"},
        "Commodity": {"commodity"},
    },
    "Produce": {
        "Flavor/Cut": {"blend", "variety", "salad mix"},
        "Shape": {
            "baby",
            "bite",
            "cob",
            "coin",
            "florette",
            "segment",
            "stick",
            "stripe",
            "wedge",
        },
        "Skin": set(),
        "Seed/Bone": {"pitted", "seedless"},
        "Processing": {
            "chopped",
            "cupped",
            "halved",
            "shredded",
            "in juice",
            "in water",
        },
        "Cooked/Cleaned": {"cleaned"},
        "WG/WGR": set(),
        "Dietary Concern": {"low sodium", "low sugar"},
        "Additives": {"no additives"},
        "Dietary Accommodation": set(),
        "Frozen": {"frozen"},
        "Packaging": {"jarred", "canned", "ss"},
        "Commodity": {"commodity"},
    },
    "Seafood": {
        "Flavor/Cut": {"loin", "meat", "shank", "steak"},
        "Shape": {
            "cake",
            "chunk",
            "fillet",
            "flake",
            "nugget",
            "patty",
            "popcorn",
            "popper",
            "portion",
            "square",
            "stick",
            "surimi",
            "rectangle",
            "ring",
            "wedge",
            "whole",
        },
        "Skin": set(),
        "Seed/Bone": {"bone-in"},
        "Processing": {
            "chopped",
            "crusted",
            "breaded",
            "battered",
            "dried",
            "in water",
            "in oil",
            "puree",
        },
        "Cooked/Cleaned": {"smoked"},
        "WG/WGR": set(),
        "Dietary Concern": {"low sodium"},
        "Additives": {"no additives"},
        "Dietary Accommodation": set(),
        "Frozen": {"frozen"},
        "Packaging": set(),
        "Commodity": {"commodity"},
    },
    "Non-Food": {
        "Bread, Grains & Legumes": set(),
        "Flavor/Cut": set(),
        "Shape": set(),
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": set(),
        "Cooked/Cleaned": set(),
        "WG/WGR": set(),
        "Dietary Concern": set(),
        "Additives": set(),
        "Dietary Accommodation": set(),
        "Frozen": set(),
        "Packaging": set(),
        "Commodity": set(),
    },
}

ADDED_GROUP_TAGS = {
    "Beverages": {
        "Flavor/Cut": set(),
        "Shape": {"ground", "kcup", "mix", "powder", "whole bean"},
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": set(),
        "Cooked/Cleaned": set(),
        "WG/WGR": set(),
        "Dietary Concern": {
            "diet",
            "caffeinated",
            "decaffeinated",
            "no sugar",
            "reduced calorie",
            "sugar free",
        },
        "Additives": {"sweetened", "unsweetened"},
        "Dietary Accommodation": set(),
        "Frozen": {"frozen", "iced"},
        "Packaging": set(),
        "Commodity": set(),
    },
    "Bread, Grains & Legumes": {
        "Flavor/Cut": set(),
        "Shape": {"cut"},
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": {"blanched"},
        "Cooked/Cleaned": {"raw", "cooked", "parboiled", "baked", "shelled"},
        "WG/WGR": set(),
        "Dietary Concern": {
            "low sodium",
            "reduced sodium",
            "low carb",
            "low sugar",
            "reduced fat",
            "low fat",
        },
        "Additives": set(),
        "Dietary Accommodation": set(),
        "Frozen": set(),
        "Packaging": {"canned"},
        "Commodity": set(),
    },
    "Condiments & Snacks": {
        "Flavor/Cut": set(),
        "Shape": {"powder", "cut", "ground", "sheet", "mix", "unsliced"},
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": {"dehydrated", "granulated"},
        "Cooked/Cleaned": {"roasted", "raw"},
        "WG/WGR": set(),
        "Dietary Concern": {
            "low fat",
            "fat free",
            "less sodium",
            "less salt",
            "low calorie",
            "low carb",
            "no sugar",
            "reduced sodium",
            "reduced sugar",
            "sugar free",
        },
        "Additives": {"salted", "sweetened", "unsalted", "unsweetened", "caffeinated"},
        "Dietary Accommodation": set(),
        "Frozen": set(),
        "Packaging": {"jarred"},
        "Commodity": set(),
    },
    "Meals": {
        "Flavor/Cut": set(),
        "Shape": {"patty", "cut"},
        "Skin": {"skin on"},
        "Seed/Bone": {"boneless"},
        "Processing": set(),
        "Cooked/Cleaned": {
            "cooked",
            "blanched",
            "raw",
            "baked",
            "parcooked",
            "parfried",
            "uncooked",
        },
        "WG/WGR": set(),
        "Dietary Concern": {"low sodium", "no salt added", "salt free"},
        "Additives": set(),
        "Dietary Accommodation": {"gluten free"},
        "Frozen": {"frozen"},
        "Packaging": {"canned"},
        "Commodity": set(),
    },
    "Meat": {
        "Flavor/Cut": set(),
        "Shape": {"sliced", "cut", "stick"},
        "Skin": {"skinless"},
        "Seed/Bone": set(),
        "Processing": set(),
        "Cooked/Cleaned": {"raw", "roast", "smoked"},
        "WG/WGR": set(),
        "Dietary Concern": set(),
        "Additives": set(),
        "Dietary Accommodation": set(),
        "Frozen": set(),
        "Packaging": {"ss", "canned"},
        "Commodity": set(),
    },
    "Milk & Dairy": {
        "Flavor/Cut": {"flavored"},
        "Shape": {"shredded", "sliced", "grated", "cut", "string", "wheel"},
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": set(),
        "Cooked/Cleaned": set(),
        "WG/WGR": set(),
        "Dietary Concern": {
            "nonfat",
            "low fat",
            "fat free",
            "light",
            "low sodium",
            "low sugar",
            "no sugar added",
            "reduced fat",
            "reduced sodium",
        },
        "Additives": {"additives", "rbst-free", "sweetened"},
        "Dietary Accommodation": set(),
        "Frozen": set(),
        "Packaging": {"canned"},
        "Commodity": set(),
    },
    "Produce": {
        "Flavor/Cut": set(),
        "Shape": {"large", "cut"},
        "Skin": {"skin on"},
        "Seed/Bone": set(),
        "Processing": set(),
        "Cooked/Cleaned": {"roasted", "cooked"},
        "WG/WGR": set(),
        "Dietary Concern": set(),
        "Additives": {"additives", "sweetened", "unsweetened"},
        "Dietary Accommodation": set(),
        "Frozen": set(),
        "Packaging": set(),
        "Commodity": set(),
    },
    "Seafood": {
        "Flavor/Cut": set(),
        "Shape": {"filet", "cubed"},
        "Skin": {"skinless"},
        "Seed/Bone": set(),
        "Processing": {"tail on", "tail off", "shell on", "peeled and deveined"},
        "Cooked/Cleaned": {"raw", "cooked"},
        "WG/WGR": {"whole grain rich"},
        "Dietary Concern": set(),
        "Additives": set(),
        "Dietary Accommodation": set(),
        "Frozen": set(),
        "Packaging": {"canned", "pouch", "ss"},
        "Commodity": set(),
    },
    "Non-Food": {
        "Bread, Grains & Legumes": set(),
        "Flavor/Cut": set(),
        "Shape": set(),
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": set(),
        "Cooked/Cleaned": set(),
        "WG/WGR": set(),
        "Dietary Concern": set(),
        "Additives": set(),
        "Dietary Accommodation": set(),
        "Frozen": set(),
        "Packaging": set(),
        "Commodity": set(),
    },
}

# FOOD PRODUCT CATEGORIES #
"""
## PRODUCE ##
Fruit
Vegetables
Roots & Tubers
## MILK & DAIRY ##
Butter
Cheese
Milk
Yogurt
Milk & Dairy (Includes other items, buttermilk, ice cream, coffee creamer, etc)
## MEAT ##
Beef
Chicken
Eggs
Pork
Turkey, Other Poultry
Meat (includes other like bison, lamb, veal, venison, etc)
## SEAFOOD ##
Fish (Farm-raised)
Fish (Wild)
Seafood (includes other)
## BREAD, GRAINS & LEGUMES ##
Grain Products
Legumes
Rice
Tree Nuts & Seeds
Bread, Grains & Legumes
"""

CATEGORY_TAGS = {
    ## GROUP: PRODUCE ##
    "Fruit": {
        "Flavor/Cut": set(),
        "Shape": set(),
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": set(),
        "Cooked/Cleaned": set(),
        "WG/WGR": set(),
        "Dietary Concern": set(),
        "Additives": set(),
        "Dietary Accommodation": set(),
        "Frozen": set(),
        "Packaging": set(),
        "Commodity": set(),
    },
    "Vegetables": {
        "Flavor/Cut": set(),
        "Shape": set(),
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": set(),
        "Cooked/Cleaned": set(),
        "WG/WGR": set(),
        "Dietary Concern": set(),
        "Additives": set(),
        "Dietary Accommodation": set(),
        "Frozen": set(),
        "Packaging": set(),
        "Commodity": set(),
    },
    "Roots & Tubers": {
        "Flavor/Cut": set(),
        "Shape": set(),
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": set(),
        "Cooked/Cleaned": set(),
        "WG/WGR": set(),
        "Dietary Concern": set(),
        "Additives": set(),
        "Dietary Accommodation": set(),
        "Frozen": set(),
        "Packaging": set(),
        "Commodity": set(),
    },
    ## GROUP: MILK & DAIRY ##
    "Cheese": {
        "Flavor/Cut": {"flavored"},
        "Shape": {
            "stick",
            "string",
            "shredded",
        },
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": {
            "dried",
            "grated",
            "sauce",
            "shaved",
            "sliced",
            "whipped",
        },
        "Cooked/Cleaned": set(),
        "WG/WGR": set(),
        "Dietary Concern": {"2%", "reduced fat", "reduced sodium"},
        "Additives": {"no additives"},
        "Dietary Accommodation": set(),
        "Frozen": {"frozen"},
        "Packaging": {"ss"},
        "Commodity": {"commodity"},
    },
    "Milk": {
        "Flavor/Cut": {"flavored"},
        "Shape": {"thickened"},
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": set(),
        "Cooked/Cleaned": set(),
        "WG/WGR": set(),
        "Dietary Concern": set(),
        "Additives": set(),
        "Dietary Accommodation": {"lactose free"},
        "Frozen": set(),
        "Packaging": {"bag", "ss"},
        "Commodity": {"commodity"},
    },
}

ADDED_CATEGORY_TAGS = {
    ## GROUP: PRODUCE ##
    "Fruit": {
        "Flavor/Cut": set(),
        "Shape": set(),
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": set(),
        "Cooked/Cleaned": set(),
        "WG/WGR": set(),
        "Dietary Concern": set(),
        "Additives": set(),
        "Dietary Accommodation": set(),
        "Frozen": set(),
        "Packaging": set(),
        "Commodity": set(),
    },
    "Vegetables": {
        "Flavor/Cut": set(),
        "Shape": set(),
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": set(),
        "Cooked/Cleaned": set(),
        "WG/WGR": set(),
        "Dietary Concern": set(),
        "Additives": set(),
        "Dietary Accommodation": set(),
        "Frozen": set(),
        "Packaging": set(),
        "Commodity": set(),
    },
    "Roots & Tubers": {
        "Flavor/Cut": set(),
        "Shape": set(),
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": set(),
        "Cooked/Cleaned": set(),
        "WG/WGR": set(),
        "Dietary Concern": set(),
        "Additives": set(),
        "Dietary Accommodation": set(),
        "Frozen": set(),
        "Packaging": set(),
        "Commodity": set(),
    },
    ## GROUP: MILK & DAIRY ##
    "Cheese": {
        "Flavor/Cut": set(),
        "Shape": set(),
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": set(),
        "Cooked/Cleaned": set(),
        "WG/WGR": set(),
        "Dietary Concern": set(),
        "Additives": set(),
        "Dietary Accommodation": set(),
        "Frozen": set(),
        "Packaging": set(),
        "Commodity": set(),
    },
    "Milk": {
        "Flavor/Cut": set(),
        "Shape": set(),
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": set(),
        "Cooked/Cleaned": set(),
        "WG/WGR": set(),
        "Dietary Concern": set(),
        "Additives": set(),
        "Dietary Accommodation": set(),
        "Frozen": set(),
        "Packaging": set(),
        "Commodity": set(),
    },
}


def create_combined_tags(level="group"):
    """Choices for level are "group" or "category"""
    if level == "group":
        tags = GROUP_TAGS
        added_tags = ADDED_GROUP_TAGS
    else:
        tags = CATEGORY_TAGS
        added_tags = ADDED_CATEGORY_TAGS
    combined_tags = {}
    for name, tags_dict in tags.items():
        combined_tags[name] = tags_dict
        all_set = set()
        for col, tag_set in tags_dict.items():
            tags_to_add = added_tags[name][col]
            combined_set = tag_set | tags_to_add
            combined_tags[name][col] = combined_set
            all_set |= combined_set
        combined_tags[name]["All"] = all_set
    return combined_tags

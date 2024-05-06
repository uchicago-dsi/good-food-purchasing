# More complicated conditional rules
# TODO: Find rules for converting flavors to "flavored"
# TODO: some of the cheese is incorrect in the dataset » or wait, this is all "milk & dairy"
# TODO: cheese types? "manchego", "mozzarella", "blue", etc.
# TODO: "blue" is a problem for colors and cheese » what about "white" for vegetables? "onion" but maybe we want this for other ones
# also white corn, etc.
# TODO: "whole weat" for pasta? ravioli?

from cgfp.config_token_map import TOKEN_MAP_DICT

SKIP_TOKENS = {
    ## GENERAL ##
    "organic",
    "breakfast",
    "superfood",
    "holiday",
    ## BEVERAGES ##
    "frappaccino",
    "frappuccino",
    "cold brew",
    "guatemala",
    ## PRODUCE ##
    "fresh",
    "with pits",
    "washed",
    ## BRAND NAMES ##
    "cheerios",
    "coke",
    "pikes place",
    "kind",
    "6th avenue bistro",
    "ybarra",
    "sriracha",  # TODO: Seems like sriracha is allowed...sometimes?
    "fantastix",
    "3 musketeer",
    "bailey's irish",
    "bailey's vanilla cream",
    ## FLAVORS (BUT DON'T TAG AS FLAVORED) ##
    "honey wheat",
    "salted caramel",
    ## BONELESS ##
    "boneless",
    ## CEREAL TYPES ##
    "chex",
    "spooners",
    "kids stuff",
    "homestyle",
    "instant",
    ## BREAD ##
    "loaf",
    "round",
    "country white",
    "old fashioned",
    "seeded",
    "sprouted",
    ## SNACKS, PASTRIES, ETC. ##
    "long john",
    ## COLORS ##
    "red",
    "light red",
    "dark red",
    "yellow",
    "green",
    "gold",
    "white",
    "blue",
    "black",
    "brown",
    "orange",  # basic type will still be set to orange since it doesn't pass through token handler
    ## DESCRIPTORS ##
    "mini",
    "snack",
    ## FRUIT ##
    "with pits",
    ## SAUSAGE TYPES ##
    "andouille",
    "polish",
    "chorizo",
    "louisiana",
    "kielbasa",
    "uncured",
    ## SORT OF FLAVORED ##
    "spicy",
    "hot and spicy",
    "glazed",
    "applewood",
    "parmesan basil",
    "extra spicy",
    "extreme heat",
    "plain",
    ## OLIVE OIL ##
    "virgin",
    "extra virgin",
    "oil blend",
    ## TEXTURE ##
    "chewie",
    "chewy",
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
    ## RICE TYPES ##
    "long grain",
    ## CHEESE ##
    "small curd",
    ## SEAFOOD ##
    "claw",
    "ahi",
    ## SIZE ##
    "jumbo",
    "king size",
    "giant",
    ## SNACKS ##
    "elf",
    "teddy",
    "bear",
    "bunny",
    "goldfish",
    ## SHAPE ##
    "hole",
    "ring",
    "twist",
    "pocket",
    "rope",
    ## SPICES ##
    "pasilla negro",
    ## TEXTURE ##
    "soft",
    "hard",
    "liquid",
    ## ETHNICITIES ##
    "Persian",
    "Cuban",
    ## MISC ##
    "sea salt",
}

# For these basic types, skip anything that is in the FLAVORS set
SKIP_FLAVORS = {
    "drink",
    "tea",
    "coffee",
    "candy",
    "condiment",
    "cereal",
    "oat",
    "bean",
    "ice cream",
    "cheesecake",
    "cracker",
    "dessert",
    "pastry",
    "cracker",
    "cookie",
    "cake",
    "danish",
    "pastry",
    "dessert",
}

# For these basic types, tag anything that includes a FLAVORS tag as "flavored"
# TODO: what happens if we return flavored twice? should probably have some deduping eventually
# Check "bread, naan, garlic, chili" to see what happes here
FLAVORED_BASIC_TYPES = {
    "bread",
    "yogurt",
    "french toast",
    "chip",
    "cranberry",
    "spread",
    "butter",
    "fruit ice",
    "popsicle",
    "mix",
}

FLAVORS = {
    ## BEVERAGES & DRINKS ##
    # sweet drinks #
    "mocha",
    # spices #
    "vanilla",
    "cinnamon",
    # misc #
    "maple",
    ## CANDY ##
    "butterscotch",
    "coffee",
    "caramel",
    "m&m",
    "toffee",
    "milk",
    ## BREAD ###
    "garlic",
    "chili",
    ## CHIPS ##
    "barbecue",
    "barbeque",
    "bbq",
    "sea salt",
    "flamin hot",
    "cheddar",
    "sour",
    "white cheddar",
    "sour cream",
    "variety",
    "extreme heat",
    "buffalo ranch",
    "queso",
    "cheddar and black pepper",
    "cheddar sour cream",
    "salt",
    "vinegar",
    "dill pickle",
    "jalapeno cheddar",
    ## CONDIMENT (SYRUP, ETC.) ##
    "maple",
    ## CEREAL ##
    "brown sugar",
    "apple cinnamon",
    "cinnamon toast crunch",
    ## LEGUMES ##
    "seasoned",
    ## CHEESE ##
    "horseradish",
    "chives",
    "sauce",
    "beer",
    ## SEAFOOD ##
    "chipotle",
    "chive and cheddar",
    "parmesan basil",
    ## HERBS ##
    "mint",
    ## SNACKS ##
    "chocolate",
}

SKIP_SHAPE = {"chip", "candy"}

SHAPE_EXTRAS = {
    ## CANDY ##
    "truffle",
    "bar",
    "bark",
    ## CHIPS ##
    "tortilla",
    "triangle",
    "ridge",
    "round",
    "ridged",
    "lattice cut",
    "popped",
    "bowl",
    "scoop",
    "crisps",
}

FRUITS = {
    "orange",
    "guava",
    "apple",
    "berry",
    "lemon",
    "lime",
    "strawberry",
    "banana",
    "raspberry",
    "passion fruit",
    "pomegranate",
    "acai",
    "blueberry",
    "cherry",
    "peach",
    "pear",
    "watermelon",
    "watermelon strawberry",
}

ALL_FLAVORS = FLAVORS | FRUITS

# TODO: Maybe dynamically generate fruits and vegetables
VEGETABLES = {"produce", "carrot", "cauliflower", "carrot", "pea", "celery", "broccoli"}

NUTS = {"almond", "cashew", "pecan", "pistachio"}

# TODO: Set this up for adding "blend" for multiple kinds of cheese
CHEESE_TYPES = {
    "cheddar",
    "monterey jack",
    "mozzarella",
    "jack",
    "provolone",
    "blue",
    "havarti",
    "gouda",
    "muenster",
    "white cheddar",
}

MELON_TYPES = {"cantaloupe", "honeydew", "watermelon"}


# TODO: is this ok?
CHOCOLATE = {"dark chocolate", "chocolate covered"}

# FOOD PRODUCT CATEGORY & GROUP STRUCTURE #
"""
Produce
Milk & Dairy
Meat
Seafood
Beverages
Meals
Condiments & Snacks
"""

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

GROUP_CATEGORY_VALIDATION = {
    "Produce": ["Fruit", "Vegetables", "Roots & Tubers"],
    "Milk & Dairy": ["Butter", "Cheese", "Milk", "Yogurt", "Milk & Dairy"],
    "Meat": [
        "Beef",
        "Chicken",
        "Eggs",
        "Pork",
        "Turkey, Other Poultry",
        "Meat",
    ],  # "Meat" is used for other items without a category like bison, lamb, venison, rabbit, etc.
    "Seafood": [
        "Fish (Farm-Raised)",
        "Fish (Wild)",
        "Seafood",
    ],  # "Seafood" is used for fish that are unconfirmed farm-raised or wild, shellfish, crab, mollusks, scallops, clams, shrimp, etc.
    "Bread, Grains & Legumes": [
        "Grain Products",
        "Legumes",
        "Rice",
        "Tree Nuts & Seeds",
        "Bread, Grains & Legumes",
    ],  # "Bread, Grains & Legumes" is used to account for items not belonging to the other categories
    "Beverages": ["Beverages"],
    "Meals": [
        "Meals"
    ],  # TODO: Figure out how to set something up for primary food product category here
    "Condiments & Snacks": ["Condiments & Snacks"],
}

GROUP_TAGS = {
    "Beverages": {
        "Flavor/Cut": {"flavored"},
        "Shape": {
            "thickened",
            "concentrate",
            "ground",
            "kcup",
            "mix",
            "powder",
            "whole bean",
            "base",
        },
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": set(),
        "Cooked/Cleaned": set(),
        "WG/WGR": set(),
        "Dietary Concern": {
            "low sodium",
            "reduced sugar",
            "diet",
            "caffeinated",
            "decaffeinated",
            "no sugar",
            "reduced calorie",
            "sugar free",
        },
        "Additives": {"no additives", "sweetened", "unsweetened"},
        "Dietary Accommodation": {"gluten free", "non-dariy", "vegan"},
        "Frozen": {"frozen/iced", "frozen", "iced"},
        "Packaging": {"ss"},
        "Commodity": {"commodity"},
    },
    "Bread, Grains & Legumes": {
        "Flavor/Cut": {"flavored"},
        "Shape": {"cut"},
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": {"bleached", "dried", "halved", "blanched"},
        "Cooked/Cleaned": {
            "oil roasted",
            "parbaked",
            "roasted",
            "raw",
            "cooked",
            "parboiled",
            "baked",
            "shelled",
        },
        "WG/WGR": {"whole grain rich"},
        "Dietary Concern": {
            "salted",
            "unsalted",
            "low sodium",
            "reduced sodium",
            "low carb",
            "low sugar",
            "reduced fat",
            "low fat",
        },
        "Additives": {"no additives", "additives"},
        "Dietary Accommodation": {"gluten free"},
        "Frozen": {"frozen"},
        "Packaging": {"ss", "canned"},
        "Commodity": {"commodity"},
    },
    "Condiments & Snacks": {
        "Flavor/Cut": {"flavored"},
        "Shape": {"powder", "cut", "ground", "sheet", "mix", "unsliced"},
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": {
            "diced",
            "dried",
            "in juice",
            "sliced",
            "dehydrated",
            "granulated",
        },
        "Cooked/Cleaned": {
            "baked",
            "cooked",
            "fried",
            "kettle cooked",
            "roasted",
            "raw",
        },
        "WG/WGR": {"whole grain rich"},
        "Dietary Concern": {
            "no sodium",
            "low sodium",
            "no sugar added",
            "reduced fat",
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
        "Additives": {
            "no additives",
            "additives",
            "salted",
            "sweetened",
            "unsalted",
            "unsweetened",
        },
        "Dietary Accommodation": {"gluten free", "kosher", "vegan", "vegetarian"},
        "Frozen": {"frozen"},
        "Packaging": {"ss", "canned", "jarred"},  # ss means "single serve"
        "Commodity": {"commodity"},
    },
    "Meals": {
        "Flavor/Cut": set(),
        "Shape": {"patty", "cut"},
        "Skin": {"skin on"},
        "Seed/Bone": {"boneless"},
        "Processing": {"breaded", "condensed", "dehydrated"},
        "Cooked/Cleaned": {
            "fried",
            "roasted",
            "cooked",
            "blanched",
            "raw",
            "baked",
            "parcooked",
            "parfried",
            "uncooked",
        },
        "WG/WGR": {"whole grain rich"},
        "Dietary Concern": {
            "reduced fat",
            "reduced sodium",
            "low sodium",
            "no salt added",
            "salt free",
        },
        "Additives": {"no additives"},
        "Dietary Accommodation": {
            "halal",
            "kosher",
            "vegan",
            "vegetarian",
            "gluten free",
        },
        "Frozen": {"frozen"},
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
            "sliced",
            "cut",
            "stick",
            "crumble",
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
        "Cooked/Cleaned": {
            "charbroiled",
            "cooked",
            "fried",
            "roasted",
            "raw",
            "roast",
            "smoked",
        },
        "WG/WGR": set(),
        "Dietary Concern": {"low sodium", "reduced sodium", "reduced fat"},
        "Additives": {"no additives"},
        "Dietary Accommodation": {"gluten free", "halal", "kosher"},
        "Frozen": {"frozen"},
        "Packaging": {"ss", "canned"},
        "Commodity": {"commodity"},
    },
    "Milk & Dairy": {
        "Flavor/Cut": {"flavored"},
        "Shape": {"shredded", "sliced", "grated", "cut", "string", "wheel"},
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": {"condensed", "evaporated", "powdered"},
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
        "Additives": {"no additives", "additives", "rbst-free", "sweetened"},
        "Dietary Accommodation": {"lactose free"},
        "Frozen": {"frozen"},
        "Packaging": {"ss", "canned"},
        "Commodity": {"commodity"},
    },
    "Produce": {
        "Flavor/Cut": {"blend", "variety", "salad mix"},
        "Shape": {
            "bite",
            "cob",
            "coin",
            "florette",
            "segment",
            "stick",
            "stripe",
            "wedge",
            "large",
            "cut",
        },
        "Skin": {"skin on"},
        "Seed/Bone": {"pitted"},
        "Processing": {
            "chopped",
            "cupped",
            "halved",
            "shredded",
            "in juice",
            "in water",
            "dried",
        },
        "Cooked/Cleaned": {"cleaned", "roasted", "cooked"},
        "WG/WGR": set(),
        "Dietary Concern": {"low sodium", "low sugar"},
        "Additives": {"no additives", "additives", "sweetened", "unsweetened"},
        "Dietary Accommodation": set(),
        "Frozen": {"frozen"},
        "Packaging": {"jarred", "canned", "ss"},
        "Commodity": {"commodity"},
    },
    "Seafood": {
        "Flavor/Cut": {"loin", "meat", "shank", "steak", "flavored"},
        "Shape": {
            "cut",
            "cake",
            "chunk",
            "filet",
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
            "cubed",
        },
        "Skin": {"skinless"},
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
            "tail on",
            "tail off",
            "shell on",
            "peeled and deveined",
        },
        "Cooked/Cleaned": {"smoked", "raw", "cooked"},
        "WG/WGR": {"whole grain rich"},
        "Dietary Concern": {"low sodium"},
        "Additives": {"no additives"},
        "Dietary Accommodation": set(),
        "Frozen": {"frozen"},
        "Packaging": {"canned", "pouch", "ss"},
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

# FOOD PRODUCT CATEGORIES #

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
            "crumble",
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
    "Yogurt": {
        "Flavor/Cut": {"flavored"},
        "Shape": {"thickened"},
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": set(),
        "Cooked/Cleaned": set(),
        "WG/WGR": set(),
        "Dietary Concern": set(),
        "Additives": set(),
        "Dietary Accommodation": {"lactose free", "gluten free"},
        "Frozen": set(),
        "Packaging": {"bag", "ss"},
        "Commodity": {"commodity"},
    },
}

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
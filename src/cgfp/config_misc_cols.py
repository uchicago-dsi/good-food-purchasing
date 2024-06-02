FPG2FPC = {
    "Produce": [
        "Fruit",
        "Vegetables",
        "Roots & Tubers",
        "Produce",
        "Legumes",
    ],  # TODO: Wait, is produce allowed here?
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

FPC2FPG = {}

for group, categories in FPG2FPC.items():
    for category in categories:
        FPC2FPG[category] = group

MISC_COLUMN_TAGS = {
    "All": {
        "Flavor/Cut": {"flavored"},
        "Shape": {"concentrate", "cut", "ground", "jerky"},
        "Skin": set(),
        "Seed/Bone": set(),
        "Processing": {"battered", "breaded", "dried"},
        "Cooked/Cleaned": {"cooked"},
        "WG/WGR": {"whole grain rich"},
        "Dietary Concern": {
            "fat free",
            "low fat",
            "low sodium",
            "nonfat",
            "reduced fat",
            "reduced sodium",
            "reduced sugar",
            "salted",
            "unsalted",
        },
        "Additives": {"additives", "no additives", "sweetened"},
        "Dietary Accommodation": {
            "gluten free",
            "halal",
            "kosher",
            "vegan",
            "vegetarian",
        },
        "Frozen": {"frozen"},
        "Packaging": {"canned", "jarred", "pouch", "ss"},
        "Commodity": {"commodity"},
    },
    ### FOOD PRODUCT GROUPS ###
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
            "wing",
            "ham",
        },
        "Shape": {
            "bacon",
            "chunk",
            "diced",
            "hot dog",
            "meatball",
            "nugget",
            "pepperoni",
            "patty",
            "salami",
            "crumble",
        },
        "Seed/Bone": {"bone-in"},
        "Processing": {"corned"},
        "Cooked/Cleaned": {"smoked"},
    },
    "Seafood": {
        "Skin": {"tail on", "shell on"},
        "Seed/Bone": {"bone-in"},
        "Cooked/Cleaned": {"smoked"},
    },
    "Condiments & Snacks": {
        "Processing": {"dehydrated", "powder"},
        "Seed/Bone": {"pitted"},
        "Dietary Accommodation": {"non-dairy"},
    },
    "Beverages": {
        "Shape": {"thickened"},
        "Dietary Concern": {"decaffeinated", "diet", "caffeinated"},
        "Frozen": {"iced"},
    },
    "Milk & Dairy": {
        "Processing": {"evaporated"},
        "Dietary Concern": {"1%", "2%"},
        "Dietary Accommodation": {"lactose free"},
    },
    "Meals": {"Processing": {"dehydrated"}},  # Note: Meals also share all "Meat" tags
    ### FOOD PRODUCT CATEGORIES ###
    "Fruit": {"Seed/Bone": {"pitted"}},
    "Cheese": {"Processing": {"grated"}},
    "Milk": {"Shape": {"thickened"}},
    "Yogurt": {"Shape": {"thickened"}},
    "Legumes": {"Processing": {"dehydrated"}},
    "Roots & Tubers": {"Processing": {"dehydrated"}},
}

# TODO: Add rule so that all "Meat" tags are added to "Meals"
NON_SUBTYPE_TAGS_FPC = {}

for fpc in FPC2FPG.keys():
    fpg = FPC2FPG[fpc]
    all_tags = set()
    NON_SUBTYPE_TAGS_FPC[fpc] = {}
    for col, tags in MISC_COLUMN_TAGS["All"].items():
        NON_SUBTYPE_TAGS_FPC[fpc][col] = tags
        fpc_tags = MISC_COLUMN_TAGS.get(fpc, {}).get(col, set())
        fpg_tags = MISC_COLUMN_TAGS.get(fpg, {}).get(col, set())
        NON_SUBTYPE_TAGS_FPC[fpc][col].update(fpc_tags | fpg_tags)
        all_tags.update(NON_SUBTYPE_TAGS_FPC[fpc][col])
    NON_SUBTYPE_TAGS_FPC[fpc]["All"] = all_tags

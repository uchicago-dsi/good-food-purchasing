"""Defines hierarchical tagging structure for CGFP tagging system"""

FPG2FPC = {
    "Produce": [
        "Fruit",
        "Vegetables",
        "Roots & Tubers",
        "Produce",
        "Legumes",
    ],
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
    "Meals": ["Meals"],
    "Condiments & Snacks": ["Condiments & Snacks"],
    "Non-Food": ["Non-Food"],
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
        "Processing": {
            "battered",
            "breaded",
            "dried",
            "in oil",
            "in water",
            "in sauce",
            "puree",
            "seasoned",
            "whipped",
            "in vegetable broth",
            "in vinegar",
        },
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
            "reduced calorie",
            "salted",
            "unsalted",
            "no sodium",
            "sugar free",
        },
        "Additives": {"additives", "no additives", "sweetened", "unsweetened"},
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
            "round",
            "butt",
            "tripe",
            "marrow bone",
            "neck",
            "rib",
            "teres major",
            "cheek",
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
            "guanciale",
            "mortadella",
            "pastrami",
        },
        "Skin": {"skin on"},
        "Seed/Bone": {"bone-in"},
        "Processing": {"corned", "stuffed"},
        "Cooked/Cleaned": {"smoked"},
    },
    "Seafood": {
        "Skin": {"tail on", "shell on", "skin on"},
        "Seed/Bone": {"bone-in"},
        "Cooked/Cleaned": {"smoked"},
    },
    "Condiments & Snacks": {
        "Processing": {"dehydrated", "powder", "in juice", "in syrup"},
        "Seed/Bone": {"pitted"},
        "Dietary Accommodation": {"non-dairy"},
    },
    "Beverages": {
        "Shape": {"thickened"},
        "Flavor/Cut": {"mix"},
        "Dietary Concern": {"decaffeinated", "diet", "caffeinated"},
        "Frozen": {"iced"},
    },
    "Milk & Dairy": {
        "Processing": {"evaporated", "powder", "grated"},
        "Dietary Concern": {"1%", "2%"},
        "Dietary Accommodation": {"lactose free"},
    },
    "Produce": {
        "Processing": {"in brine"},
    },
    ### FOOD PRODUCT CATEGORIES ###
    "Fruit": {"Seed/Bone": {"pitted"}, "Processing": {"in juice", "in gel", "in syrup", "in light syrup"}},
    "Cheese": {"Shape": {"crumble"}},
    "Milk": {"Shape": {"thickened"}},
    "Yogurt": {"Shape": {"thickened"}},
    "Legumes": {"Processing": {"dehydrated"}},
    "Roots & Tubers": {"Processing": {"dehydrated"}},
    "Eggs": {"Processing": {"hard boiled"}},
    "Pork": {"Shape": {"iberico"}},
}

# Note: Tags are aggregated on the Food Product Category level so we can easily check if a tag
# is not a subtype. We create a dictionary for allowed tags for each Food Product Category by
# combining tags that are allowed for all items and tags allowed for the Food Product Group
# Note: Meals are allowed to have tags from the Primary Food Product Category. This is handled
# in the name normalization process.
NON_SUBTYPE_TAGS_FPC = {}

for fpc in FPC2FPG.keys():
    fpg = FPC2FPG[fpc]
    all_tags = set()
    NON_SUBTYPE_TAGS_FPC[fpc] = {}
    for col, tags in MISC_COLUMN_TAGS["All"].items():
        # Start with tags allowed for all items
        # Note: "whipped" is not allowed for Meals or Meat
        if col == "Processing" and (fpg == "Meals" or fpg == "Meat"):
            tags = tags.copy()  # Make a copy so we don't modify the original
            tags.remove("whipped")
        NON_SUBTYPE_TAGS_FPC[fpc][col] = tags
        # Add tags for the Food Product Group & Food Product Category
        fpg_tags = MISC_COLUMN_TAGS.get(fpg, {}).get(col, set())
        fpc_tags = MISC_COLUMN_TAGS.get(fpc, {}).get(col, set())
        NON_SUBTYPE_TAGS_FPC[fpc][col].update(fpc_tags | fpg_tags)
        # Keep track of all allowed tags for easy checking on whether to save a tag as subtype
        all_tags.update(NON_SUBTYPE_TAGS_FPC[fpc][col])
    # Update all allowed tags for Food Product Category
    NON_SUBTYPE_TAGS_FPC[fpc]["All"] = all_tags

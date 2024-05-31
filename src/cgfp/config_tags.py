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
    "whole bean",
    ## PRODUCE ##
    "fresh",
    "with pits",
    "washed",
    ## BRAND NAMES ##
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
    "cheerios",  # TODO: wait, some cereal names are allowed?
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
    "grey",
    ## DESCRIPTORS ##
    "mini",
    "snack",
    ## PRODUCE ##
    "with pits",
    "cob",
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
    "large",
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
    "sheet",
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
    ## OLD MISC COLUMNS TAGS ##
    "cleaned",
    "kettle cooked",
    "raw",  # Assume products are raw unless labeled otherwise
    "shelled",
    "unsliced",
    "wheel",
    "bite",
    "skinless",
    "bleached",
    "cupped",
    "granulated",
    "tail off",
    "peeled and deveined",
    "light",
    "low calorie",
    "low carb",
    "low sugar",
    "rbst-free",
    ## MISC SUBTYPES ##
    "4 way",
    "5 spice",
    "5 way",
    "7 layer",
    "7 up",
    "7 way",
    "active",
    "adzuki",
    "alaea",
    "al pastor",
    "alpha bits",
    "alphabet",
    "apple leather",
    "azure blue",
    "back",
    "baker",
    "ball",
    "beauty heart",
    "beer battered",
    "bell shaped",
    "belt",
    "bi-color",
    "bias",
    "bite-size",
    "bitter",
    "black and white",
    "blackening",
    "blending",
    "butterfinger",
    "by the foot",
    "chick o stick",
    "chunky",
    "clear",
    "coated",
    "coating",
    "cortland",
    "costa rica",
    "costa rican",
    "cotswold",
    "diet coke",
    "dinosaur",
    "dirty",
    "dover",
    "ear",
    "easter",
    "easter egg",
    "exotic",
    "extra lean",
    "extra sweet",
    "fajita strip",
    "feather",
    "feet",
    "fine",
    "flamas",
    "flaming hot",
    "flat iron",
    "forbidden",
    "food",
    "foretrotter",
    "frapuccino",
    "frose rose",
    "frosted",
    "froth",
    "fruit by the foot",
    "frying",
    "fudge striped",
    "full rib",
    "funyun",
    "galactic green",
    "gentile",
    "german butterball",
    "ghost blend",
    "gigante",
    "glaze",
    "glazed with tabasco",
    "globe",
    "gold bar",
    "gold ginger",
    "gold rush",
    "golden fancy",
    "golden graham",
    "golden nugget",
    "gordita",
    "gournay",
    "granbury gold",
    "grape berry",
    "grape leather",
    "green breaker",
    "guajillo chili",
    "gypsy",
    "h&r blend",
    "haas",
    "half cob",
    "hawaiian punch",
    "head",
    "head off",
    "heart shaped",
    "heath",
    "heath bar",
    "heavy syrup",
    "heel",
    "heel meat",
    "hero",
    "hershey",
    "high gluten",
    "himalayan",
    "hind shank",
    "hindshank",
    "hock",
    "holiday variety",
    "home fry",
    "homestyle",
    "no sauce",
    "no top",
    "no tops",
    "peruvian",
    "picnic",
    "reeses",
    "reeses and cream",
    "reeses peanut butter cup",
    "restaurant style",
    "runts",
    "rum and date",
    "sea",
    "stalk",
    "taki",
    "takis",
    "tidbits",
    "tiger",
    "tiger striped",
    "titdbit",
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

# GROUP_CATEGORY_VALIDATION = {
#     "Produce": ["Fruit", "Vegetables", "Roots & Tubers"],
#     "Milk & Dairy": ["Butter", "Cheese", "Milk", "Yogurt", "Milk & Dairy"],
#     "Meat": [
#         "Beef",
#         "Chicken",
#         "Eggs",
#         "Pork",
#         "Turkey, Other Poultry",
#         "Meat",
#     ],  # "Meat" is used for other items without a category like bison, lamb, venison, rabbit, etc.
#     "Seafood": [
#         "Fish (Farm-Raised)",
#         "Fish (Wild)",
#         "Seafood",
#     ],  # "Seafood" is used for fish that are unconfirmed farm-raised or wild, shellfish, crab, mollusks, scallops, clams, shrimp, etc.
#     "Bread, Grains & Legumes": [
#         "Grain Products",
#         "Legumes",
#         "Rice",
#         "Tree Nuts & Seeds",
#         "Bread, Grains & Legumes",
#     ],  # "Bread, Grains & Legumes" is used to account for items not belonging to the other categories
#     "Beverages": ["Beverages"],
#     "Meals": [
#         "Meals"
#     ],  # TODO: Figure out how to set something up for primary food product category here
#     "Condiments & Snacks": ["Condiments & Snacks"],
# }

# MISC_COLUMN_TAGS = {
#     "All": {
#         "Flavor/Cut": {"flavored"},
#         "Shape": {"concentrate", "cut", "ground", "jerky"},
#         "Skin": set(),
#         "Seed/Bone": set(),
#         "Processing": set(),
#         "Cooked/Cleaned": set(),
#         "WG/WGR": set(),
#         "Dietary Concern": set(),
#         "Additives": set(),
#         "Dietary Accommodation": set(),
#         "Frozen": set(),
#         "Packaging": set(),
#         "Commodity": set(),
#     },
#     "Meat": {
#         "Flavor/Cut": {
#             "brisket",
#             "breast",
#             "loin",
#             "shank",
#             "skirt",
#             "steak",
#             "short rib",
#             "t-bone",
#             "thigh",
#             "wing",
#             "ham",
#         },
#         "Shape": {
#             "bacon",
#             "chunk",
#             "diced",
#             "fillet",
#             "ground",
#             "hot dog",
#             "meatball",
#             "nugget",
#             "pepperoni",
#             "patty",
#             "pulled",
#             "salami",
#             "strip",
#             "taco meat",
#             "tender",
#             "cut",
#             "crumble",
#         },
#         "Skin": set(),
#         "Seed/Bone": set(),
#         "Processing": set(),
#         "Cooked/Cleaned": set(),
#         "WG/WGR": set(),
#         "Dietary Concern": set(),
#         "Additives": set(),
#         "Dietary Accommodation": set(),
#         "Frozen": set(),
#         "Packaging": set(),
#         "Commodity": set(),
#     },
# }

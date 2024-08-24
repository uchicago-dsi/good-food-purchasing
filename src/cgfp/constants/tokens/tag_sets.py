"""Define sets of tags that have special rules for token mapping for CGFP pipeline"""

# More complicated conditional rules
# TODO: Find rules for converting flavors to "flavored"
# TODO: some of the cheese is incorrect in the dataset » or wait, this is all "milk & dairy"
# TODO: cheese types? "manchego", "mozzarella", "blue", etc.
# TODO: "blue" is a problem for colors and cheese » what about "white" for vegetables? "onion" but maybe we want this for other ones
# also white corn, etc.
# TODO: "whole weat" for pasta? ravioli?

### FLAVORS ###
# Tags to potentially be categorized as "flavored"
FLAVORS = {
    ## BEVERAGES & DRINKS ##
    # sweet drinks #
    "mocha",
    "hazelnut and cream",
    "hazelnut cream",
    "mocha latte",
    "arnold palmer",
    # sports drinks #
    "arctic blitz",
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
    "moosetracks",
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
    "glacier freeze",
    ## TEA FLAVORS ##
    "orange and spice",
    "orange blossom",
    "orange carrot",
    "orange cherry grape",
    "orange citrus",
    "orange cream",
    "orange dulce",
    "orange guava passion",
    "orange mango",
    "orange medley",
    "peach ginger",
    ## FRUIT FLAVORS ##
    "guava pear",
    "guava strawberry",
    "mixed berry",
    "mixed fruit",
    "orange pineapple cherry",
    "orange pineapple crème",
    "paradise punch",
    "passion fruit pineapple",
    "passion orange",
    "passion orange guava",
    "peach hibiscus",
    "peach mango",
    "peach pineapple",
    "peach tea",
    "pear ginger",
    "yumberry pomegranate",
    "wildberry",
    "wild cherry",
    "strawberry and cream",
    "strawberry beet",
    "strawberry champagne",
    "strawberry chocolate",
    "strawberry cream",
    "strawberry crisp",
    "strawberry crème",
    "strawberry dragonfruit",
    "strawberry gelatin",
    "strawberry kiwi",
    "strawberry lemon",
    "strawberry lemonade",
    "strawberry mango",
    "strawberry oatmeal",
    "strawberry orange",
    "strawberry peach",
    "strawberry pomegranate",
    "strawberry raspberry",
    "strawberry rhubarb",
    "strawberry rosemary",
    "strawberry shortcake",
    "strawberry vanilla",
    "strawberry watermelon",
    "strawbery kiwi",
    "tangerine mango guava",
    "tangerine orange",
    "watermelon agave",
    "watermelon flavor",
    "watermelon passion fruit",
    "watermelon passionfruit",
    "wild berry",
    "white grape peach",
    "white grape cherry",
    ## CANDY FLAVORS ##
    "sour apple",
    "sour ball",
    "star blast",
    ## MISC ##
    "herb and garlic",
    "honey barbecue",
    "honey barbecue glaze",
    "honey barbeque",
    "honey brown sugar",
    "mango chili lime",
    "mango hibiscus",
    "mango lime",
    "mango peach",
    "mango pineapple",
    "mango sticky rice",
    "mango strawberry pomegranate",
    "peanut butter chocolate butterscotch",
    "peanut butter crème",
    "wild white nacho",
    "spicy nacho",
    "spicy sichuan",
    "spicy sweet",
    "sweet and salty",
    "sweet orange",
    "sweet potato chocolate",
    "sweet potato chocolate chip",
    "sweet salty",
    "sweet spicy",
    "toasted almond",
    "toasted cinnamon",
    "tropical",
    "tropical banana",
    "tropical blend",
    "tropical citrus",
    "tropical fire",
    "tropical fruit",
    "tropical island",
    "tropical mango",
    "tropical orange",
    "tropical passion",
    "tropical punch",
    "tropical twist",
    "turmeric aloe",
    "vanilla almond",
    "vanilla cola",
    "vanilla wafer",
    "vanilla cream",
    "vanilla crème",
    "vanilla fig",
    "vanilla honey",
    "vanilla latte",
    "vanilla m&m",
    "vanilla milk chocolate",
    "vanilla strawberry",
    "vanilla caramel",
    "vanilla chocolate",
    "white chocolate chip",
    "sour cherry",
    "sour cherry lemon",
    "sour cream and onion",
    "sour heads",
    "sour patch",
    "sour patch kid",
    "sour patch kids",
    "sour raspberry",
    "sour watermelon",
}

# Note: fruits are separated since there is separate logic for "fruit, blend"
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
    "coconut",
    "grapefruit",
    "mango",
    "blackberry",
}

ALL_FLAVORS = FLAVORS | FRUITS

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
    "cookie",
    "cake",
    "danish",
}

# For these basic types, tag anything that includes a FLAVORS tag as "flavored"
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

# TODO: This ends up incorrect since it has two flavors
#  YOGURT ASSORTED RASPBERRY/PEACH L/F G/F	yogurt, rapsberry, peach, low fat	yogurt, rapsberry, flavored, low fat

### SHAPE ###

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

### BLENDS/VARIETY/ETC ###
# These are tags that should be relabeled if there are multiple tags from the same category

# TODO: Maybe dynamically generate fruits and vegetables
VEGETABLES = {
    "produce",
    "carrot",
    "cauliflower",
    "pea",
    "celery",
    "broccoli",
    "pepper",
    "onion",
    "green bean",
    "wax bean",
    "edamame",
}

NUTS = {"almond", "cashew", "pecan", "pistachio"}

CHEESE_TYPES = {
    "cheddar",
    "cheddar cheese",
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

MELON_TYPES = {"cantaloupe", "honeydew", "watermelon", "galia"}

SUBTYPE_REPLACEMENT_MAP = {
    "fruit": "fruit",
    "cheese": "blend",
    "vegetable": "blend",
    "melon": "variety",
}

### MISC ###
CHOCOLATE = {"dark chocolate", "chocolate covered"}

CORN_CERAL = {
    "frosted corn flakes",
    "frosted flake",
    "frosted flakes",
    "honey nut chex",
    "trix",
}

RICE_CEREAL = {"special k"}

WHEAT_CEREAL = {
    "frosted mini spooner",
    "frosted mini spooners",
    "frosted mini wheat",
    "frosted mini wheats",
    "frosted spooners",
    "frosted wheat",
    "frosted wheats",
    "fruit whirls",
    "kashi",
    "mini spooners",
    "mini spooner",
    "wheat bran flakes",
    "wheat frosted",
    "wheaties",
}

OAT_CEREAL = {
    "fruity cheerios",
    "cheerios",
    "honey nut cheerios",
    "honey scooters",
    "lucky charms",
    "multigrain o",
    "multigrain oats",
    "toasted oats",
    "toasted white chocolate",
    "toasty o",
}

FRUIT_SNACKS = {
    "fruit roll up",
    "fruit roll-up",
    "fruit rolls",
    "fruit rollup",
    "fruit snack",
    "gusher",
}

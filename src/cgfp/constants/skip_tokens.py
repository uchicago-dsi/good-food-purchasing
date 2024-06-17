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
    "fettuccine",
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
    "flake",
    "shaved",
    "whole",
    "no sugar",
    "no sugar added",
    "sugar free",
    "link",
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
    "guajillo chile",
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
    "1.5%",
    "100% fruit",
    "100% juice",
    "12 grain",
    "9-grain",
    "7-grain",
    "acini di pepe",
    "addis",
    "berry rain",
    "better burger",
    "cholesterol free",
    "escape",
    "fortified",
    "four grain",
    "freeze dried",
    "french blend",
    "french bread",
    "french breakfast",
    "french butter",
    "french roast",
    "french round",
    "fries",
}

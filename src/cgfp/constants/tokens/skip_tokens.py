"""Defines a set of tokens to skip during CGFP pipeline. Note that Basic Type isn't filtered through this."""

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
    "salted caramel",  # TODO: why not? salted caramel is a flavor for yogurt
    ## BONELESS ##
    "boneless",
    ## CEREAL TYPES ##
    "kids stuff",
    "homestyle",
    "homeystyle",
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
    "white",  # TODO: What about white wine? White vinegar?
    "blue",
    "black",
    "brown",
    "orange",  # basic type will still be set to orange since it doesn't pass through token handler
    "grey",
    ## DESCRIPTORS ##
    "mini",
    "snack",
    ## PRODUCE ##
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
    "gemelli",
    "radiatore",
    "torchiette",
    "elbow",
    "elbow macaroni",
    "bowtie",
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
    "rotini",
    "shell",
    "spaetzle",
    "spaghetti",
    "stuffed shell",
    "tortellini",
    "vermicelli",
    "linguine",
    "linguini",
    "ziti",
    "orecchiette",
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
    "medium",
    "tiny",
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
    "devined",  # type for "deveined"
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
    "fruit and nut in yogurt",
    "fruit swirl",
    "full fat",
    "garlic flake",
    "ginger beer",
    "ginger gold",
    "ginger spice",
    "gingerbread",
    "gingersnap",
    "alegria",
    "cotton",
    "devil's food",
    "arcadian harvest",
    "no salt added",
    "no salt",
    "salt free",
    "green tip",
    "greens",
    "hawaiian",
    "herbs",
    "hickory",
    "honey bun",
    "honey crisp",
    "honey cured",
    "honey fire",
    "honey ginger",
    "honey maple",
    "honey nut",
    "honey o",
    "honey pepper",
    "honey punch",
    "honey sunshine",
    "husked",
    "iced coffee blend",
    "iced latte",
    "iced tea",
    "icy charge",
    "in gelatin",
    "in pod",
    "jelly belly",
    "jolly rancher",
    "junior mints",
    "light brown",
    "light roast",
    "light ss",
    "lightly salted",
    "little gem",
    "madagascar vanilla",
    "main street jack",
    "malt-o-meal",
    "malted milk powder",
    "malted milkshake",
    "manos de dios",
    "marble",
    "marble jack",
    "margarita agave",
    "margarita strawberry",
    "marrow",
    "master blend",
    "meat",
    "mediterranean",
    "medium grain",
    "mexican blend",
    "mexican salt",
    "meyer",
    "miami cola",
    "micro",
    "middle",
    "mike & ike",
    "mike and ike",
    "mike and ikes",
    "milky way",
    "millet and chia",
    "miso ginger",
    "monterey",
    "moon rock silver",
    "moon skin",
    "mountain roast",
    "multi-color",
    "munchie mix",
    "munchies",
    "nature valley",
    "navy",
    "navy pea",
    "neon blast",
    "neon worm",
    "nest",
    "netted",
    "neutral",
    "new mexico",
    "no preservatives",
    "no pulp",
    "north star",
    "northern",
    "nut free",
    "nutrigrain",
    "nutrition",
    "ocean",
    "on the cob",
    "orange spice",
    "orange tangerine",
    "oreo",
    "outside",
    "pad",
    "pan",
    "parkerhouse",
    "passion",
    "patty pan",
    "peach apricot",
    "peanut butter and cheese",
    "peanut butter cup",
    "peanut butter jelly",
    "peanut caramel",
    "peanut free",
    "pearl",
    "pearled",
    "pee wee",
    "peppermint s'mores",
    "holland",
    "artichoke parmesan",
    "arlis",
    "arbol",
    "arabica",
    "almond joy",
    "african nectar",
    "zucchini bread",
    "zucchini carrot",
    "zesty",
    "yo dots",
    "ya",
    "with top",
    "with corn",
    "with backbone",
    "winter",
    "wine and beer",
    "wildflower",
    "wild swan",
    "white rose",
    "pepsi",
    "persian",
    "power c machine",
    "prince edward isle",
    "prince edward",
    "rye sourdough",
    "premium roast",
    "preserved",
    "red bull",
    "sliced whole grain rich",
    "smore bar",
    "snickers",
    "soy-free",
    "star",
    "star brite",
    "star ruby",
    "starburst",
    "straw",
    "strawberry sundae",
    "swedish fish",
    "sweet green tea",
    "sweet italian",
    "sweetango",
    "sweetened",
    "swiss fish",
    "szechwan",
    "three musketeer",
    "toasted",
    "toasted oat",
    "toasted sesame",
    "triple berry",
    "triple chocolate",
    "triple chocolate fudge",
    "unbaked",
    "v8",
    "vitamin",
    "vitamin d machine",
    "white and dark",
    "white corn",
    "white chocolate raspberry",
    "southwest",
    "southwestern",
    "apple nut",
    "apple fritter",
    "apple cinnamon",
    "apple cinnamon pecan",
    "apple cinnnamon",
    "apple cranberry",
    "apple peach",
    "apple smoked",
    "apple strawberry",
    "apple strawberry banana",
    "apple walnut",
    "apple whirls",
    "apache blue",
    "anti-browning solution",
    "animal",
    "amber",
    "americano",
    "snickerdoodle",
    "tiramisu",
    "sourdough",
    "rye",
    "oat bran",
    "oatmeal cream pie",
    "mousse",
    "black forest",
    "faming hot",
    "muscle",
    # BEAN TYPES
    "cannellini",
    "fava",
    "great northern",
    "red bean",
    "red kidney",
    "fanta",
    # BRANDS
    "pop tart",
    "poptart",
    "vitamin water",
    "wheat thin",
    # CANDY
    "candy cane",
    "skittle",
    # EXTRA SUBTYPES
    "crumb",
    "deckle-off",
    "fletch",
    "funnel",
    "bottle",
    "apple caremel",
    "baja blast",
    "bakes",
    "basted",
    "berry red",
    "blue hubbard",
    "blue machine",
    "boil in bag",
    "bomb pop",
    "breakfast blend",
    "brown spicy",
    "brownie turtle",
    "bugle",
    "cacciatore",
    "caffe verona",
    "canada dry",
    "canadian style",
    "cashew cookie",
    "chamomile lemon",
    "chinese",
    "chinese five spice",
    "christmas",
    "clif",
    "cobb",
    "colombian",
    "colored",
    "country",
    "country style",
    "creole style",
    "crispy",
    "crunchy",
    "crustless",
    "cushion",
    "dark",
    "dark meat",
    "dark roast",
    "deli",
    "denuded",
    "destemmed",
    "dr. pepper",
    "drained",
    "drop",
    "dunkers",
    "dutch",
    "edible",
    "english breakfast",
    "everything",
    "extra fine",
    "extra firm",
    "extra light",
    "extra wide",
    "femur",
    "five blend",
    "five way",
    "french toast shape",
    "freshly prepared",
    "fruit smoothie",
    "fruity",
    "fry",
    "gallon",
    "german style",
    "gourmet",
    "graham crust",
    "green machine",
    "gummy bear",
    "gummy worm",
    "half gallon",
    "harvest",
    "health",
    "honey toasted",
    "hot & spicy",
    "house blend",
    "hungarian",
    "israeli",
    "italian blend",
    "italian style",
    "italian wedding",
    "jamaica",
    "jamaican",
    "jamaican style",
    "japanese",
    "jerk caribbean",
    "kewpie",
    "kiev",
    "kool ranch",
    "korean",
    "kung pao",
    "latin style",
    "liberty brew",
    "lighty",
    "loaded baked potato",
    "madagascar",
    "maple crust",
    "medium firm",
    "microwavable",
    "mild",
    "mission",
    "mongolian",
    "navel valencia",
    "new england",
    "no bean",
    "no msg",
    "no oil",
    "original",
    "part skim",
    "pike place",
    "pixie",
    "pizza blend",
    "pizza style",
    "portside blend",
    "power punch",
    "primal",
    "pure",
    "quart",
    "quick",
    "ready to serve",
    "reconstituted",
    "red skin",
    "reduced fat",
    "reese's pieces",
    "rendered",
    "reuben",
    "rockit",
    "rolo",
    "ruby red",
    "saint louis",
    "salt water taffy",
    "santa claus",
    "santa fe",
    "savory",
    "scandinavia",
    "scandinavian blend",
    "schoolboy",
    "scooby doo",
    "scooters",
    "semi sweet",
    "semi-sweet",
    "seven grain",
    "shamrock",
    "shaped",
    "sharp",
    "shelf stable",
    "sierra mist",
    "sleepytime",
    "smile",
    "smooth",
    "solution added",
    "spicy szechuan",
    "stickless",
    "stroganoff",
    "sunrise",
    "sweet and sour",
    "tail",
    "thick & chunky",
    "thick crust",
    "thin",
    "thin crust",
    "three grain",
    "tri-color",
    "triangle",
    "triple delight",
    "tub",
    "tuscan",
    "two layer",
    "ultra grain",
    "uncooked",
    "unglazed",
    "unpeeled",
    "unrefined",
    "unseasoned",
    "veranda blend",
    "vienna",
    "voodoo gumbo",
    "wedding",
    "wellington",
    "western",
    "western style",
    "white frosted",
    "white nacho",
    "wide",
    "xango",
    "yolk",
    "cherry vanilla",
    "dr pepper",
    "ginger ale",
    "mountain dew",
    "oolong",
    "calrose",
    "buttered",
    "dipperdoodle",
    "empire",
    "envol",
    "fagiola pasta",
    "four way",
    "french orange",
    "fresno",
    "garnet",
    "herbed",
    "ice",
    "jamwich",
    "knot",
    "maui",
    "opal",
    "pinwheel",
    "plate",
    "purple top",
    "red machine",
    "rice crisp",
    "root beer barrel",
    "round inside",
    "smokey",
    "spray",
    "table blend",
    "tahitian breeze",
    "tango",
    "trim",
    "twisted",
    "wafel",
    "white bean cassoulet",
    "supreme",
    "anaheim",
    "arcadian",
    "ball tip",
    "black turtle",
    "boat",
    "boston",
    "bowl",
    "broken",
    "cameo",
    "cap",
    "cara cara",
    "caraway",
    "chef",
    "chioggia",
    "chub",
    "inside",
    "golden parisien",
    "hot ring",
    "prince william",
    "pullman",
    "rich",
    "riserva",
    "rod",
    "royale",
    "ruby",
    "sidewinder",
    "sport",
    "summer",
    "sweet chili",
    "table",
    "tenders",
    "texas",
    "top",
    "turtle",
    "vegetable flavor",
    "pekin",
    "livewire",
    "katlyn",
    "flagolet",
    "hostess",
    "hula cooler",
    "lemonhead",
    "bubblemint",
    "skittles",
    "blue lake",
    "carroteeni",
    "ceylon",
    "water added",
    "volcano",
    "ureka",
    "tamayokucha",
    "st. louis style",
    "redrific",
    "rainforest select",
    "tri color",
    "wheat free",
    "with gravy",
    "100%",
    "3-grain",
    "4%",
    "5 grain",
    "7 grain",
    "passover",
    "pod",
    "kona blend",
    "kidney",
    "pinto",
    "noi",
}

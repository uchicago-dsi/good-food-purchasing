"""Defines a mapping dictionary for typos and inconsistencies in CGFP data pipeline"""

# TODO: Maybe I should pull out some of the common mappings like "cut" and include some sort of set membership check
TOKEN_MAP_DICT = {
    ## TYPOS ##
    "talpia": "tilapia",
    "talapia": "tilapia",
    "omlete": "omelette",
    "omelete": "omelette",
    "grapefuit": "grapefruit",
    "cheedar": "cheddar",
    "carribean": "caribbean",
    "cardamon": "cardamom",
    "gerkin": "gherkin",
    "rapsberry": "raspberry",
    "granualted": "granulated",
    "furikaki": "furikake",
    "fettuccini": "fettuccine",
    "fettucine": "fettuccine",
    "fettucini": "fettuccine",
    "early gray": "earl grey",
    "whole grain rich  rich": "whole grain rich",
    "orchiette": "orecchiette",
    "campanelli": "campanelle",
    "unsweeted": "unsweetened",
    "peeled & deveined": "peeled and deveined",
    "mangu": "mango",
    "siced": "sliced",
    "apple golden delicious": "apple",
    "apple. Gala": "apple",
    "apple. gala": "apple",  # TODO: apple type is allowed so handle these differently
    "bicsuit": "biscuit",
    "brussel": "brussels sprout",
    "brussel spout": "brussels sprout",
    "brussel sprout": "brussels sprout",
    "bulghur": "bulgur",
    "bulgar": "bulgur",
    "bulgar wheat": "bulgur",
    "cabbabe": "cabbage",
    "collard": "collard greens",
    "collard green": "collard greens",
    "freekah": "freekeh",
    "grean pea": "green pea",
    "grean bean": "green bean",
    "green brean": "green bean",
    "grit": "grits",
    "jicima": "jicama",
    "lemon grass": "lemongrass",
    "onio": "onion",
    "pummelo": "pomelo",
    "rasberry": "raspberry",
    "shelll": "shell",
    "tamato": "tomato",
    "1 island": "1000 island",
    "alfafa": "alfalfa",
    "brocolini": "broccolini",
    "code": "cod",
    "berry lemonage": "berry lemonade",
    "gryo meat": "gyro meat",
    "patry": "pastry",
    "sweetner": "sweetener",
    "tat soi": "tatsoi",
    "tostado shell": "tostada",
    "chery": "cherry",
    "grean": "green",
    "green chili": "green chile",
    "guajillo chili": "guajillo chile",
    "habenero": "habanero",
    "hambuger": "hamburger",
    "harvarti": "havarti",
    "honey mustard dijon": "honey mustard",
    "iceburg": "iceberg",
    "jonogold": "jonagold",
    "marianra": "marinara",
    "mayonaise": "mayonnaise",
    "mayonaisse": "mayonnaise",
    "mirepoux": "mirepoix",
    "monterey jark": "monterey jack",
    "montery jack": "monterey jack",
    "mozzerella": "mozzarella",
    "peperoncini": "pepperoncini",
    "peppercini": "pepperoncini",
    "powedered": "powdered",
    "raisin brain": "raisin bran",
    "reeses pieces": "reese's pieces",
    "sandwhich": "sandwich",
    "snickerdoogle": "snickerdoodle",
    "springle": "sprinkle",
    "sqare rib": "spare rib",
    "stawberry": "strawberry",
    "sun-dried": "sun dried",
    "tumeric": "turmeric",
    "yogurt alternertive": "yogurt alternative",
    "tobasco": "tabasco",
    "burrio": "burrito",
    "raddicchio": "radicchio",
    "chcolate": "chocolate",
    "decaffienated": "decaffeinated",
    "decaffinated": "decaffeinated",
    "decafienated": "decaffeinated",
    # TODO: red ti leaf
    ## INCONSISTENCIES ##
    "cheeks": "cheek",
    "cheez it": "cheez-it",
    "cheez-its": "cheez-it",
    "non-fat": "nonfat",  # TODO: which do we want to do? Make sure to change the misc tags
    "skin-on": "skin on",
    "carrots": "carrot",
    "gluten-free": "gluten free",
    "tail-on": "tail on",
    "tail-off": "tail off",
    "sugar-free": "sugar free",
    "fillet": "filet",
    "fat-free": "fat free",
    "rice crispies": "rice krispies",
    "bread crumb": "breadcrumb",
    "bread stick": "breadstick",
    "brussels sprouts": "brussels sprout",
    "bulgur wheat": "bulgur",
    "cranberries": "cranberry",
    "egg": "eggs",
    "heart of palm": "hearts of palm",
    "peppers": "pepper",
    "vegetables": "vegetable",
    "vetegable": "vegetable",
    "wheat bulgur": "bulgur",
    "yuca": "yucca",
    "beef steak": "beefsteak",
    "corn meal": "cornmeal",
    "cornstarch": "corn starch",
    "egg nog": "eggnog",
    "micro greens": "microgreen",
    "micro green": "microgreen",
    "pimiento": "pimento",
    "preserves": "preserve",
    "spring": "spring mix",
    "tostada shell": "tostada",
    "turnip green": "turnip greens",
    "additive": "additives",
    "chives": "chive",
    "corn puffs": "corn puff",
    "cornish game hen": "cornish hen",
    "cous cous": "couscous",
    "dougnut": "doughnut",
    "florets": "floret",
    "froot loop": "froot loops",
    "fruit loop": "froot loops",
    "fruit loops": "froot loops",
    "frito": "fritos",
    "grannysmith": "granny smith",
    "cocoa": "cocoa powder",
    "grapes": "grape",
    "entree": "entrée",
    "chili relleno": "chile relleno",
    "relleno chile": "chile relleno",
    "beverage": "drink",
    ## COOKED ##
    "au jus": "cooked",
    "charbroiled": "cooked",
    "oil roasted": "cooked",
    "roast": "cooked",
    "autumn roasted": "cooked",
    "blanched": "cooked",
    "char-broiled": "cooked",
    ## RENAME ##
    "aprium": "pluot",
    "banana pepper": "pepper",
    "blood orange": "orange",
    "bluelake bean": "bean",
    "bouillon": "base",
    "cotton candy": "candy",
    "pig": "pork",
    "romano bean": "bean",
    "pigeon pea": "bean",
    "cracker meal": "breadcrumb",
    "hasu": "lotus root",
    "satsuma": "mandarin",
    "white rice": "rice",
    ## CUT ##
    "sliced": "cut",
    "diced": "cut",
    "chopped": "cut",
    "wedge": "cut",
    "segment": "cut",
    "baby": "cut",
    "julienne": "cut",
    "julienned": "cut",
    "quartered": "cut",
    "quarter": "cut",
    "cubed": "cut",
    "chunk": "cut",
    "trimmed": "cut",
    "half": "cut",
    "halved": "cut",
    "spear": "cut",
    "crinkle cut": "cut",
    "florette": "cut",
    "filet": "cut",
    "portion": "cut",
    "rectangle": "cut",
    "coin": "cut",
    "square": "cut",
    "strip": "cut",
    "stripe": "cut",
    "tender": "cut",
    "airline breast": "cut",
    "deli sliced": "cut",
    "fillet diced": "cut",
    ## WHOLE GRAIN ##
    "whole wheat": "whole grain rich",
    "whole grain": "whole grain rich",
    "white whole wheat": "whole grain rich",
    ## COOKED ##
    "baked": "cooked",
    "fried": "cooked",
    "roasted": "cooked",
    "parboiled": "cooked",
    "parcooked": "cooked",
    "broiled": "cooked",
    "parbaked": "cooked",
    "parfried": "cooked",
    "blackened": "cooked",
    "flame broiled": "cooked",
    "flamebroiled": "cooked",
    ## FLAVORED ##
    "clear blue raspberry": "flavored",
    "frost riptide": "flavored",
    "garden salsa": "flavored",
    "garden vegetable": "flavored",
    "garlic parmesan": "flavored",
    "garlic roasted and herb": "flavored",
    "ginger lemon": "flavored",
    "ginger peach": "flavored",
    "glacier cherry": "flavored",
    "glacier freeze": "flavored",
    "glacier ice": "flavored",
    "blueberry vanilla": "flavored",
    ## PROCESSING ##
    "condensed": "concentrate",
    "crusted": "breaded",
    "powdered": "powder",
    ## PACKAGING ##
    "bag": "ss",
    "kcup": "ss",
    ## IN JUICE ##
    "in pear juice": "in juice",
    ## SMOKED ##
    "pecanwood smoked": "smoked",
    ## MIXED VEGGIES ##
    "vegetable blend": "vegetable",  # TODO: maybe we need this in basic type handler
    "vegetable cup": "vegetable",
    ## PASTA TYPES ##
    "penne rigate": "pasta",
    ## PASTRIES ##
    "sweet roll": "pastry",
    ## DIETARY CONCERN ##
    "less sodium": "low sodium",
    "less salt": "low sodium",
    ## CAFFEINE ##
    "decaf": "decaffeinated",
    ## SUB-TYPE INCONSISTENCIES ##
    "ancho chile powder": "ancho chile",
    "apple oatmeal": "oatmeal",
    "apple pie": "pie",
    "blackeye pea": "pea",
    "cheese filled": "cheese",
    "chicken fried": "fried",
    "chili ancho": "ancho chile",
    "cracker sandwich": "cracker",
    "earl gray": "earl grey",
    "egg white": "egg",
    "french catalina": "catalina",
    "french cut": "cut",
    "french honey": "french",
    "french red": "french",
    "fresco rancher": "queso fresco",
    "garlic pepper steak": "garlic pepper",
}

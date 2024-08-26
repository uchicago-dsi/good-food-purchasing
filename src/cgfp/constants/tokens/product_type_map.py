"""Defines product type mappings for edge cases in CGFP tagging system"""

PRODUCT_TYPE_MAP = {
    "TOMATO HEIRLOOM CHERRY MIX 12 PT": {"Basic Type": "tomato", "Sub-Types": ["cherry"]},
    "HAWAIIAN PUNCH CANS": {"Basic Type": "drink", "Sub-Types": ["fruit punch"]},
    "Concentrate Chipotle S/O": {"Basic Type": "chipotle", "Shape": "concentrate"},
    "OREGANO AQUARESIN (7.5LB/PL)": {
        "Basic Type": "spice",
        "Sub-Types": ["oregano"],
        "Shape": "concentrate",
    },
    "GRAIN, BLEND COUSCOUS TRI-COLOR QUINOA RESEALABLE BAG": {
        "Basic Type": "quinoa",
        "Sub-Types": ["couscous", "blend"],
    },
    "GREEN, MUST CHPD DMSTC IQF FZN": {
        "Basic Type": "mustard green",
        "Shape": "cut",
        "Frozen": "frozen",
    },
    "Turkey Cranberry Snack Sticks": {"Basic Type": "turkey", "Sub-Types": ["jerky"]},
    "TURKEY STICK SMOKEHOUSE": {"Basic Type": "turkey", "Sub-Types": ["jerky"]},
    "APPETIZER, CHICKEN COCONUT SKEWER .85 OZ COOKED FROZEN KABOB": {
        "Basic Type": "chicken",
        "Sub-Types": ["kabob", "coconut"],
        "Cooked/Cleaned": "cooked",
        "Frozen": "frozen",
    },
    "APPETIZER, CHICKEN SKEWER .8 OZ PARCOOKED FROZEN": {
        "Basic Type": "chicken",
        "Sub-Types": ["kabob"],
        "Cooked/Cleaned": "cooked",
    },
    "WG RAMEN MISO NOODLE KIT": {
        "Basic Type": "meal kit",
        "Sub-Types": ["noodle", "miso"],
        "WG/WGR": "whole grain rich",
    },
    "EDAMAME SUCCOTASH BLEND": {
        "Food Product Group": "Produce",
        "Food Product Category": "Vegetables",
        "Primary Food Product Category": "Vegetables",
        "Basic Type": "vegetable",
        "Sub-Types": ["blend", "edamame", "succotash"],
    },
    "MIX CAKE BASE RICHCREME": {"Basic Type": "dessert", "Sub-Types": ["cake", "mix"]},
    "QUICK OATS TUBES": {"Basic Type": "oat", "Sub-Types": ["quick"]},
    "RICE PILAF CHICKEN W/ORZO": {
        "Basic Type": "entree",
        "Sub-Types": ["chicken", "rice pilaf"],
    },
    "GRAIN SPCLTY BULGHUR WHEAT PLF": {
        "Basic Type": "bulgur",
        "Sub-Types": ["wheat", "pilaf"],
    },
    "POLENTA, CAKE VEG FIRE RSTD (6455314)": {
        "Basic Type": "entree",
        "Sub-Types": ["polenta", "vegetable"],
    },
    "ROOT TARO MALANGA COCA": {"Basic Type": "taro"},
    "SHORTBREAD STRAWBERRY": {
        "Basic Type": "dessert",
        "Sub-Types": ["shortbread", "strawberry"],
    },
    "SCOOBY DOO GRAHAM STIX IW": {
        "Basic Type": "cracker",
        "Sub-Types": ["graham"],
        "WG/WGR": "whole grain rich",
        "Packaging": "ss",
    },
    "BEAN, GOURMET MADAGASCAR BOURB (4648664)": {
        "Basic Type": "spice",
        "Sub-Types": ["vanilla bean"],
    },
    "PATE-PORK PISTACHIO OLYM 6/8 OZ": {
        "Basic Type": "pate",
        "Sub-Types": ["pork"],
    },
    "PATE-PORK RILLETE OLYM 6/8 OZ": {
        "Basic Type": "pate",
        "Sub-Types": ["pork"],
    },
    "ORANGE, BLOOD CNCNT FZN": {
        "Basic Type": "orange",
        "Shape": "concentrate",
        "Frozen": "frozen",
    },
    "GRAIN, WHEAT RED BRRY": {
        "Basic Type": "wheat berry",
    },
    "BENEFIT BRFST BAR OAT/RSN IW": {
        "Basic Type": "bar",
        "Sub-Types": ["oat"],
        "Packaging": "ss",
    },
}

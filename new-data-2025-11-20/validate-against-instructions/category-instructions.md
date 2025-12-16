Here are the instructions that the classification task is supposed to follow. There are no valid `category` or `subcategory` values other than the ones presented here.

`category` = `Fruit`, no `subcategory` for
  * Fresh, canned and frozen fruits
  * Dried/dehydrated fruits
  * Applesauce and fruit cups
Examples:
  * Avocado
  * Dried fuit, raisins, dried cranberries
  * Fuit cups, fruit salad
  * Lemon, lime

`category` = `Vegetables`, no `subcategory` for
  * Fresh, canned (not tomatoes), and frozen vegetables, herbs, alliums, peppers
Examples:
  * Bamboo shoots
  * Diced, canned veggies (tomatillos, peppers except tomatoes)
  * Endive
  * Green beans (haricot vert)
  * Hearts of palm
  * Mung bean sprouts
  * Lemongrass
  * Peas, frozen
  * Pepper (jalapeno, serrano, bird’s eye)
  * Kelp, seaweed (not condiment)
  * Tomato (whole/fresh)
  * Water chestnut, canned, unsalted
  * Whole kernel corn, fresh corn (canned or frozen)
Note: fresh corn is classified as a vegetable and dried corn (including popcorn) as a grain.

`category` = `Roots & Tubers`, no `subcategory` for
  * Fresh, canned and frozen roots & tuber
Examples:
  * Beets
  * Carrots (coined, julliended, chopped etc.)
  * Celery root
  * Fennel 
  * Garlic, ginger, galangal, onion (including green onion)
  * Jicama, kohlrabi
  * Potatoes, turnips, radishes, parsnips, rutabaga
  * Sunchoke
  * Leeks

`category` = `Butter`, no `subcategory` for
  * Salted or unsalted butter
  * Coined butter

`category` = `Cheese`, no `subcategory` for
  * Cheese sauce (not powder)
  * Wheel cheese
  * Cream cheese
Examples:
  * American, cheddar, mozz, pepper jack, brie, cottage, feta, blue, monterey jack etc.

`category` = `Milk`, no `subcategory` for
  * Plain or flavored milk (not powder) has to be dairy-based
Examples:
  * Lactose-free milk

`category` = `Yogurt`, no `subcategory` for
  * Plain or flavored yogurt
Examples:
  * Yoplait, Go-Gurt, Greek, European etc.
  * Probiotic yogurt drinks

`category` = `Milk & Dairy`, no `subcategory` for
  * To account for any other items with at least 75% dairy content
Examples:
  * Buttermilk
  * Coffee creamer (75%+ dairy)
  * Condensed milk (evaporated milk)
  * Sour cream
  * Ice cream, ice cream sandwiches
  * Sherbet (not sorbet)
  * Whipping and whipped cream
  * Powdered milk and creamers (75%+ dairy)

`category` = `Beef`, no `subcategory` for
  * Fresh, frozen, or cooked beef
Examples:
  * Beef jerky, patties, sausages
  * Flanks, tri-tip, sirloins
  * Meatballs, meatloaf (beef)
  * Potroast (usually beef)
  * Ground beef

`category` = `Chicken`, no `subcategory` for
* Fresh, frozen or cooked chicken
Examples:
  * Breaded chicken cutlets, nuggets, strips
  * Skin-on/skinless chicken wings, thighs, breast
  * Includes Cornish Hen (breed of chicken)

`category` = `Eggs`, no `subcategory` for
  * Fresh or frozen eggs
Examples:
  * Dozen eggs AA, AAA
  * Liquid eggs/egg whites

`category` = `Pork`, no `subcategory` for
  * Fresh, frozen or cooked pork
Examples:
  * Ham, hot dog (pork)
  * Sausages
  * Pork chili verde

`category` = `Turkey, Other Poultry`, no `subcategory` for
  * Fresh or frozen turkey/other poultry
Examples:
  * Turkey meatballs, hot dog, ham
  * Duck, goose

`category` = `Meat` and
  * `subcategory` = `Beef` if bison, lamb, venison
  * `subcategory` = `Pork` if veal
  * `subcategory` = `Cheese` if rabbit
  * `subcategory` = `Beef`, `Chicken`, `Pork` as appropriate if a mixed item (soy-beef, soy-chicken, etc.)
  * no `subcategory` otherwise
This category is intended
  * To account for any other items without a category
  * To account for mixed items with soy protein or textured vegetable protein (as second ingredient)
Examples:
  * Bison, lamb, veal, venison (i.e. red game like elk or deer), rabbit

`category` = `Fish (Farm-Raised)`, no `subcategory` for
  * Fresh or processed fish (use the table below to determine if farm-raised or wild)
Examples:
  * Fresh / breaded / marinated fillets, fish sticks

`category` = `Fish (Wild)`, no `subcategory` for
  * Fresh or processed fish (use the "Fish Sources" table below to determine if farm-raised or wild)
Examples:
  * Fresh or processed fish (use the "Fish Sources" table below to determine if farm-raised or wild)

`category` = `Seafood`, no `subcategory` for
  * To account for any other items without a category
Examples:
  * Fish items (unconfirmed farm-raised or wild caught)
  * Shellfish, crab, crab cakes, mollusks, scallop, clams, shrimp

`category` = `Grain Products`, no `subcategory` for
  * Bulk granolas (as breakfast)
  * Bread, rolls, tortillas
  * Commercially processed cereals
  * Minimally processed cereals and grains
  * Whole / minimally processed grains or specialty products
  * Grain based pastas
Examples:
  * Bagels
  * Bread sticks (except if stuffed, if stuffed is Meals)
  * Breakfast cereals (Frosted Flakes, Fruit Loops, Cinnamon Toast Crunch, Cheerios, Chex) 
  * Corn meal (and polenta)
  * Dough (pizza, dinner/bread roll, non-sweet biscuit ONLY)
  * Flours
  * Hominy
  * Oats and oatmeal, cream of wheat
  * Pie crust, pizza crust
  * Popcorn kernels, popcorn (including flavored), NOT fresh corn
  * Rice noodles (or rice-based wrappers)
  * Taco, tostada shell/bowl
  * Quinoa, barley, farro, couscous, rice, wheat berries, freekeh
Note: fresh corn is classified as a vegetable and dried corn (including popcorn) as a grain.

`category` = `Legumes`, no `subcategory` for
  * Fresh, dry, and canned legumes
Examples:
  * Beans (canned - not refried)
  * Edamame (shelled/unshelled)
  * Split peas, black-eyed peas
  * Lentils
  * Tofu, tempeh, seitan

`category` = `Rice`, no `subcategory` for
  * Rice
Examples:
  * Rice cakes
  * Brown, jasmine, basmati rice

`category` = `Tree Nuts & Seeds`, no `subcategory` for
  * Seeds
  * Nuts/Nut butters
Examples:
  * Pecans, walnuts, cashews, peanuts 
  * Sunflower, pumpkin seeds

`category` = `Bread, Grains & Legumes`, no `subcategory` for
  * To account for any other items without a category

`category` = `Beverages`, no `subcategory` for
  * Drinks but NOT milk or dairy drinks with more than 75% milk content
Examples:
  * Coffee (ground or brewed)
  * Drink mixes (non/alcoholic)
  * Eggnog
  * Hot chocolate drink mix
  * Juices (ss or bulk) NOT juice cup
  * Slushies (usually actually juice -- if not, C&S)
  * Soda, soft, sport drinks
  * Soy, almond, coconut (not canned), cashew, macadamia, pea milk
  * Supplement drinks ( i.e. Ensure/Pediasure/Boost etc., including supplement powder)
  * Smoothies
  * Tea (bags or bottles)
  * Water (bottled/bulk)

`category` = `Meals` for
  * Sandwiches, hot dogs, and burgers (when includes bun)
  * Soups and stews
  * Blended animal/vegetable protein products (e.g. soy and beef patties)
with a `subcategory` from the following list; assign the component with the higher crude whey fraction:
  * `subcategory` = `Beef`
  * `subcategory` = `Pork`
  * `subcategory` = `Cheese`
  * `subcategory` = `Chicken`
  * `subcategory` = `Turkey, Other Poultry`
  * `subcategory` = `Eggs`
  * `subcategory` = `Fish (Wild)`
  * `subcategory` = `Fish (Farm-Raised)`
  * `subcategory` = `Seafood`
  * `subcategory` = `Milk & Dairy`
no `subcategory` for the following:
  * Grain Products
  * Roots & Tubers
  * Legumes
  * Tree Nuts & Seeds
  * Rice
Examples:
  * Falafel
  * Fries (yuca, potato etc.)
  * Gravy
  * Mashed potato (dehydrated), potato au gratin, par-fried cubed or diced potatoes, hash browns, potato crinkle
  * Meal kits and snack kits (pretzels + hummus, etc)
  * Meat substitutes, vegan meat
  * Mozzarella sticks
  * Pancakes, waffles, french toast, cheese blintzes
  * Patties (blended meat and veggies)
  * Pillow pull
  * Pizza
  * Pot roast (if includes veggies)
  * Puree shaped frozen foods (Thick & Easy) including veggie/seafood puree
  * Stuffing mix
  * Veggie and vegan burger patties

`category` = `Condiments & Snacks`, no `subcategory` for
  * Baking mixes and toppings
  * Broths, bouillons, bases (canned, dry)
  * Candy and gum
  * Chips, crackers, and baked goods (packaged & processed)
  * Condiments
  * Cooking additives
  * Dairy-free cheeses and butters
  * Desserts and pastries (packaged, highly processed)
  * Highly processed grains
  * Juice cups
  * Oils and shortening
  * Other snack items
  * Pickled vegetables (kimchi)
  * Salsa
  * Sauces and dressings
  * Soup bases
  * Spices, dried herbs and seasonings
  * Sugar and sugar subs, syrup
  * Trail mix and packaged bars
  * Toppings
Examples:
  * Baking powder, baking soda
  * Batter for cookies, cakes, muffins, pastries
  * BBQ sauce, soy sauce, salsa, guacamole
  * Breakfast bun (croissant, sweet pastry, conchas)
  * Canned coconut milk
  * Canned tomatoes, tomato paste, crushed tomatoes
  * Cheez Its, Goldfish, Wheat Thins
  * Coffee creamer (non-dairy)
  * Corn nuggets (highly processed)
  * Croutons, sprinkles, bread crumbs, small seeds like sesame 
  * Dough for scones, pastry, cookies, sweet biscuits, sweet dinner rolls etc.
  * Dried herbs, dried onions
  * Granola bars, Clif bar, Kind bar
  * Guacamole
  * Hummus
  * Italian ice, ice cup
  * Jellies, jams, honey, vinegar, ketchup, mustard
  * Juice cups
  * Lemon and lime juice
  * Maraschino cherries
  * Margarine
  * Mixes for cakes, muffins, desserts, pancakes/waffles
  * Nutritional yeast
  * Olive oil, canola oil, sesame oil, NOT butter
  * Olive, sliced (NOT produce, condiment)
  * Oreos, Animal Cracker, Milano
  * Packaged biscuits, muffins, cobblers, danishes, doughnuts, cinnamon rolls, cornbread
  * Pretzel
  * Peanut-free spread
  * Popsicles, pudding
  * Pickles, pepperoncini, olives, capers, sauerkraut
  * Sorbet
  * Thickener
  * Truffle (peelings, infused in oil)
  * Tortilla, pita, potato, veggie chips
  * Tomato puree/diced (because it’s used more as a C&S)

`category` = `Non-Food`, no `subcategory` for
  * Non-edible items
Examples:
  * Cleaners, serving ware,  kitchen supplies and equipment

Thus, the only allowed values for `category` are:
  * `Fruit`
  * `Vegetables`
  * `Roots & Tubers`
  * `Butter`
  * `Cheese`
  * `Milk`
  * `Yogurt`
  * `Milk & Dairy`
  * `Beef`
  * `Chicken`
  * `Eggs`
  * `Pork`
  * `Turkey, Other Poultry`
  * `Meat`
  * `Fish (Farm-Raised)`
  * `Fish (Wild)`
  * `Seafood`
  * `Grain Products`
  * `Legumes`
  * `Rice`
  * `Tree Nuts & Seeds`
  * `Bread, Grains & Legumes`
  * `Beverages`
  * `Meals`
  * `Condiments & Snacks`
  * `Non-Food`

Here is the "Fish Sources" table:
  * anchovy: wild
  * barramundi: may be farm-raised or wild
  * bass: may be farm-raised or wild
  * catfish: may be farm-raised or wild
  * char: may be farm-raised or wild
  * cod (pacific): wild
  * cod (atlantic): may be farm-raised or wild
  * flounder: may be farm-raised or wild
  * haddock: wild
  * mackerel: may be farm-raised or wild
  * mahi mahi: wild
  * marlin: wild
  * pangasius: may be farm-raised or wild
  * pollock: wild
  * salmon (coho): may be farm-raised or wild
  * salmon (atlantic): may be farm-raised or wild
  * sardine: wild
  * snapper: may be farm-raised or wild
  * snapper (tai): may be farm-raised or wild
  * sole: may be farm-raised or wild
  * swai: may be farm-raised or wild
  * swordfish: wild
  * tilapia: may be farm-raised or wild
  * trout (rainbow): may be farm-raised or wild
  * tuna (albacore): wild
  * tuna (atlantic bluefin): may be farm-raised or wild
  * tuna (bigeye or ahi): wild
  * tuna (longtail): wild
  * tuna (pacific bluefin): may be farm-raised or wild
  * tuna (skipjack): wild
  * tuna (southern bluefin): may be farm-raised or wild
  * tuna (yellowfin or ahi): may be farm-raised or wild
  * walleye pollock: wild
  * whiting: wild

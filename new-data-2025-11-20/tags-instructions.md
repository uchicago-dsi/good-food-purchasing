Here are the instructions that the tagging task is supposed to follow.

General rules:
  * tags are always lowercase
  * obvious attributes, such as "frozen" for "ice cream" or "vegetarian" for "bean & cheese" should not be explicit
  * each attribute should be singular, like "strawberry", not "strawberries"
  * always use correct spelling
  * for boolean-valued tags, missing tags are equivalent to `false`

Overarching pattern for tags: each tag follows a pattern that informs which attributes are relevant to include and the order in which to place them. While not all items will have product tags, the tags aim to answer five questions:
  * Q1: What are the major item descriptors/ingredients?
  * Q2: What are the physical attributes of the product?
  * Q3: How has the item been chemically or physically processed?
  * Q4: What dietary attributes does the product have?
  * Q5: How is the item stored?

Q1: What are the major item descriptors/ingredients?
  * `basic_type`: the simplest term to describe the most basic element of the product, such as "juice", "snack", or "beef".
  * `sub_types`: descriptors that inform the products primary, secondary, and tertiary characteristics, if available. About half of the food products have a primary sub-type, 1 in 10 have a secondary sub-type, and 1 in 100 have a tertiary sub-type.

Q2: What are the physical attributes of the product?
  * `flavored`: if the product has added flavor, this value is `true`; otherwise, it is `false` or absent
  * `shape`: shape or consistency, possible values if present: `cut`, `patty`, `ground`, `concentrate`, `bacon`, `hot dog`, `meatball`, `thickened`, `crumble`, `nugget`, `jerky`, `salami`, `pepperoni`, `pastrami`, `bologna`, `prosciutto`, `shredded`, `genoa`, `liquid`, `mortadella`, `capocollo`, `pancetta`, `bresaola`, `sopressata`, `breast`, `cotto`, `guanciale`, `nostrano`
  * `meat_cut`: for meat only, possible values are: `breast`, `ham`, `wing`, `thigh`, `steak`, `loin`, `rib`, `mix`, `tenderloin`, `leg`, `brisket`, `chuck`, `butt`, `sirloin`, `shoulder`, `short rib`, `bottom round`, `belly`, `shank`, `oxtail`, `skirt`, `tri tip`, `striploin`, `knuckle`, `rack`, `shortloin`, `cheek`, `neck`, `round`, `tripe`, `tongue`, `teres major`, `pectoral meat`, `outside skirt`, `marrow bone`, `loin rib`, `t-bone`, `cut`
  * `meat_skin`: for meat only, may be `skin on`, `tail on`, or `shell on`; otherwise, it is absent
  * `meat_bone`: for meat only, if it contains bone, this value is `true`; otherwise, it is `false` or absent
  * `seed_pitted`: for foods that can have seeds but have been pitted (seed removed), this value is `true`; otherwise, it is `false` or absent

Q3: How has the item been chemically or physically processed?
  * `processing`: if processed or preserved in some way, this value is `breaded`, `in juice`, `seasoned`, `dried`, `in syrup`, `puree`, `powder`, `in water`, `battered`, `hard boiled`, `dehydrated`, `whipped`, `grated`, `corned`, `in sauce`, `stuffed`, `in brine`, `in oil`, `evaporated`, `in puree`, `in vinegar`, `in liquid`, `in gel`, `marinated`, `powdered`, `in vegetable broth`; otherwise, it is absent
  * `cooked`: if cooked or smoked, this value is `cooked` or `smoked`; otherwise, it is absent

Q4: What dietary attributes does the product have?
  * `whole_grain`: if whole grain rich (WG or WGR), this value is `true`; otherwise, it is `false` or absent
  * `fat_content`: may be `nonfat`, `low fat`, `1%`, `2%`, `fat free`, or absent
  * `sodium_level`: may be `low sodium`, `reduced sodium`, `salted`, `unsalted`, or absent
  * `caffeine`: may be `decaffeinated`, `caffeinated`, or absent
  * `diet`: `diet`, `reduced calorie`, or absent
  * `reduced_sugar`: may be `true`, `false`, or absent
  * `sweetened`: may be `sweetened`, `unsweetened`, or absent
  * `additives`: may be `additives`, `no additives`, or absent
  * `dietary_accommodation`: may be `gluten free`, `kosher`, `vegan`, `vegetarian`, `lactose free`, `halal`, `non-dairy`, or absent

Q5: How is the item stored?
  * `frozen`: may be `frozen`, `iced`, or absent if not frozen in any way
  * `packaging`: may be `ss` for single-serve, `canned`, `jarred`, or absent
  * `commodity`: if the item is a commodity purchase (i.e. brown box, divertided, USDA, FFVP), then this value should be `true`; otherwise, it is `false` or absent

Collapsed terms that apply to all food product categories:
  * Commodity:
    * examples: `CMDY`, `USDA`; collapsed term: `commodity` = `true`
  * Dietary accommodation:
    * `VG`; collapsed term: `vegan`
    * `GF` or `GLTN FR`; collapsed term: `gluten free`
    * `OU` or `O-U`: collapsed term: `kosher`
  * Dietary attributes:
    * `LF`, `low-fat`, `L/F`; collapsed term: `low fat`
    * `UNSWN`, `SF` (sugar free); collapsed term: `sweetened` = `false`
    * `LOW-S`; collapsed term: `low sodium`
    * `RED SOD`; collapsed term: `reduced sodium`
    * `SLTD` or `SALTD`; collapsed term: `salted`
    * `UNSLTD` or `NSA`; collapsed term: `unsalted`
  * Freezing:
    * `FZ`, `IQF`, `FRZ`, `FRZN`; collapsed term: `frozen`
  * Packaging:
    * `SS`, `BOWL`, `BWL`, `PACKET`, `PC`, `IW`, `IND`, `POUCH`, `BULKPAK`, `BLKPK`; collapsed term: `ss` for single-serve
    * `IN JAR`, `JAR`; collapsed term: `jarred`
  * Whole grain rich:
    * `WG`, `WGR`, `WHL GRAIN`; collapsed term: `whole_grain` = `true`
  * `RS` can either mean `reduced_sugar` = `true` or `reduced sodium`, depending on the item. For example:
    * `WG RS APPLE JACKS CEREAL` is `reduced_sugar` = `true` (cereal doesn't typically have a lot of salt)
    * `WG RS MAC AND CHEESE LG ELBOW` is `reduced sodium` (mac and cheese doesn't typically have a lot of sugar)
  * `LS` can either mean `low sodium`, low sugar (and so `sweetened` = `false`), or light syrup (and so `processing` = `in syrup`). For example:
    * `GRAVY MIX BROWN PAN RS LS` is `low sodium`
    * `FRUIT COCKTAIL LS` is light syrup
  * `LSS` is large serving size (ignore)
  * `WP` is wet pack (ignore)

Notes for each major type of food product

Beverages:
  * `basic_type`
    * milk alternatives (nut, soy, oat, etc.), always `plant milk`
    * juice slushies and juice cups are `juice`
    * lime and lemon juice are `juice`
    * canned coconut milk is `plant milk`
    * powdered mixes, concentrates (input contains `BIB`), and K Cups (including mixes for lemonade, "juices", etc.) are `drink`
  * `sub_types`
    * kind of plant milk, kind of juice, etc.
    * do not include soda type
    * do not include brand name
    * `cocktail` if less than 100% juice (e.g. 15% cranberry juice has `basic_type` = `juice`, `sub_types` = `["cranberry cocktail"]`)
  * beverage descriptor
    * chamomile is assumed `herbal`
    * coffee is assumed caffeinated; set `caffeine` = `decaffeinated` if specified
    * do not include `100%` for apple or orange juice
  * note if `flavored`
  * note consistency: `shape` = `thickened` or `concentrate`
  * note processing: `mix` for concentrates, mixes, and syrups
  * note dietary attributes
  * note additives
  * note dietary accommodation
    * only when not obvious (i.e. orange juice is always vegan, so it doesn't need to be noted, and berry smoothie is assumed to be dairy unless otherwise stated)
  * note frozen or iced
  * note packaging
    * `MINI`, `1/2 PT`, `4-8 OZ`, `CARTON`/`CTN`, `BOTTLE` are all `SS` (single-serve)
  * note commodity

Breads, grains, and legumes:
  * `basic_type`
    * the word `grits` is singular; do not shorten to `grit`
  * note if `flavored`
  * note processing
    * grain products are assumed `cut`, so it doesn't need to be noted
  * note if `whole_grain`
  * note dietary concern
  * note additives
  * note dietary accommodation
  * note frozen
  * note packaging
  * note commodity
  
Cheese:
  * `basic_type` is always `cheese`
  * note cheese type
  * note if `flavored`
  * note shape
    * generally `stick` or `string`
  * note processing
    * generally whether the cheese was cut
  * note dietary concern
    * generally the sodium or fat content
    * list fat content as shown in product description
  * note additives
  * note dietary accommodation
    * generally not applicable
    * all cheese is assumed gluten-free
  * note frozen
  * note packaging
    * less than 1 Oz, cups, packets are `SS` (single-serve)
  * note commodity

Condiments & Snacks:
  * `basic_type`
    * `snack` is a `basic_type` with these `sub_types`: `chex mix`, `cheese bite`, `crisp`, `corn nut`, `puff`, `fruit cup`, `fruit roll-up`, `munchies`, `popcorn`, `poptart`, `pretzel`, `rice cake`, `rice krispies`, etc.
    * cheese alternatives are `cheese substitute`
  * note if `flavored`
  * note processing
    * `SAUERKRAUT` is assumed to be cut
    * `SPICE` and `SEASONING` are assumed to be ground
    * if `PICKLES` are coins or chips, they are `cut`
  * note if `whole_grain`
  * note dietary concern
  * note additives
  * note dietary accommodation
  * note frozen
  * note packaging
    * less than 3 Oz, `PC` (portion controlled) are `SS` (single-serve)
    * `6-10` or `6/#10` are `canned`
  * note commodity

Seasonings:
  * `basic_type` = `seasoning` are spice or herb blends used to season a specific meat or dish
    * `sub_types` may be `cajun`, `chicken`, `fajita`, `poultry`, `taco`, etc.
  * `basic_type` = `spice` is a seed, fruit, root, bark, salt, or other plant substance primarily used for flavoring or coloring food
    * `sub_types` may be `allspice`, `garam masala`, `pepper`, `chili`, `cinnamon`, etc.
  * `basic_type` = `herb` is the leaves, flowers, or stems of plants used for flavoring or as a garnish
    * `sub_types` may be `basil`, `oregano`, `sage`, `thyme`, etc.

Pastries versus deserts:
  * `basic_type` = `pastry` are baked goods that are typically eaten as a breakfast or snack item
    * `sub_types` may be `banana bread`, `churro`, `cinnamon roll`, `coffee cake`, `cream puff`, `crescent`, `croissant`, `danish`, `doughnut`, `eclair`, `pan dulce`, `phyllo`, `scone`, `strudel`, `turnover`, etc.
    * this does *not* include muffins
  * `basic_type` = `dessert` are cake or pie items that are intended as a dessert item
    * `sub_types` may be `brownie`, `cake`, `cupcake`, `fudge`, `gelatin`, `lemon bar`, `pie`, `pudding`, `tart`, etc.
    * this does *not* include cookies

Meals:
  * `basic_type`
    * meat substitutes are `meatless` (see below)
  * `sub_types` indicate the main protein
    * keep any secondary `sub_types` as simple as possible, i.e. `turkey` and not `turkey bologna`
  * note processing
  * note if `cooked`
  * note if `whole_grain`
  * note dietary concerns
  * note additives
  * note dietary accommodation
  * note packaging
  * note commodity

Meat substitutes
  * `basic_type` = `meatless`
    * `sub_types` is either the meat it's replacing or the type of replacement; for example:
      * `BEEF SUB GRND MED MEATL PEA` has `sub_types` = `["beef"]`
      * `BURGER BLACK BEAN SOY VGN` has `sub_types` = `["black bean", "soy", "patty"]`
      * `VEGGIE BURGER CALIFORNIA GTF` has `sub_types` = `["vegetable", "patty"]`

Meat & Eggs:
  * for `basic_type` = `eggs`,
    * note the species in `sub_types` if not chicken
  * note that `bacon`, `ham`, and `gyro` are the `shape` of meat, not a `basic_type`
  * note the meat species
    * for blended meats
    * list in order of CWF impact
  * do not include "Angus" or "CAB" (Certified Agnus Beef)
  * `sausage` is a `basic_type`, not the `shape`
  * note `meat_cut`
  * note `shape`
  * note `meat_bone`
  * note `processing`
    * flavored or marinated `processing` = `marinated`
    * bacon is assumed to be cut
    * "pulled" meat is `meat_cut` = `cut`
    * chunks or squares are `meat_cut` = `cut`
  * note if `cooked`
  * note dietary concerns
  * note additives
  * note dietary accommodation
    * especially `kosher` or `halal`
  * frozen can be assumed
  * note commodity

Collapsed terms and common abbreviations for meats:
  * species
    * `AB` (All Beef); collapsed term: `beef`
    * `TRKY`; collapsed term: `turkey`
  * shape
    * `FL`, `FIL`; collapsed term: `filet`
    * `BRST`; collapsed term: `breast`
    * `PTY`; collapsed term: `patty`
    * `WHL`; collapsed term: `whole`
  * processing
    * `BRD`; collapsed term: `breaded`
  * cooked
    * `PRCK`, `PCK`, `FC`, `SMKD`; collapsed term: `cooked`
  * `MSC` means "Mechanically Separated Chicken"; no need to include in `basic_type` or `sub_types`

Milk:
  * `basic_type` is always `milk`
  * note if `flavored`
  * note consistency
  * note dietary accommodation
    * `lactose free`
  * note `fat_content`
    * `SKIM`, `SKM`, `FF`, `FAT FREE`, `NF`, `SK` are all `nonfat`
    * `1%`, `LF`, `LOW FAT` are all `1%`
    * `RED FAT`, `REDUCED FAT`, `RF` are all `2%`
    * `WHL` is whole (do not note `fat_content`)
  * note packaging
    * `MINI`, `1/2 PT`, `HPT`, `3/8 OZ`, 4-8 Oz, `HP`, `PT`, `PINT`, 50+/cs are all `SS` (single-serving)

Milk & Dairy
  * `frozen yogurt` is a `basic_type` not `yogurt` with `sub_types` = `["frozen"]`
  * `sub_types`:
    * half & half is `basic_type` = `creamer`, `sub_types` = `["half and half"]`
    * heavy cream is `basic_type` = `cream`, `sub_types` = `["whipping"]`
  * note if `flavored`
  * note processing
    * `condensed`, `powdered`, or `evaporated` for `milk`
    * butter is assumed to be solid
  * dietary concern
    * for yogurt: `low fat` or `light`
  * note additives
  * note dietary accommodation
  * note frozen
  * note packaging
    * for butter, `QUARTERS` means `SS` (single-serving)
    * ice cream cups, bars, and sandwiches are assumed to be single-serving
    * yogurt: 4-6 Oz or `1/2 PT` are `SS` (single-serving)
    * creamer: `3/8 OZ` or `CUP` are `SS` (single-serving)
  * note commodity

Note that `PARFAIT PRO` is a brand of bulk yogurt, not a parfait.

Produce (Fruits, Vegetables, Roots & Tubers):
  * `sub_types`
    * in a fruit cup, include the species; e.g. `basic_type` = `fruit cup`, `sub_types` = `["strawberry"]`
  * blend, mixed, variety, salad mix
    * use `basic_type` = `blend`, not `mix` (except for salad mix) because mixes are typically for powder or concentrate
    * CSA box: `basic_type` = `produce`, `sub_types` = `["variety"]`
    * spring/salad mix: `basic_type` = `lettuce`, `sub_types` = `["salad mix"]`
    * canned mixed fuits: `basic_type` = `fruit`, `sub_types` = `["variety", "canned"]`
    * fruit cup mixed: `basic_type` = `fruit cup`, `sub_types` = `["variety"]`
  * note shape
    * trimmed and shaped produce: `cut`
    * for carrots, "baby": `cut`
  * note processing
  * note dietary concern
  * note additives
  * note dietary accommodation
  * note frozen
  * note packaging
    * `PORTION`, `PRTN` are `SS` (single-serving)
    * `CAN` is `canned`
  * note commodity
 
 Some examples:
   * `basic_type` = `orange`, `sub_types` may be `navel`, `cara cara`, `valencia`
   * `basic_type` = `mandarin`, `sub_types` may be `clementine`, `tangerine`, `satsuma`, `tangelo`, `minneola`, `honeybell`
   * `basic_type` = `apple`, `sub_types` may be `granny smith`, `red`, `honey crisp`, `fuji`, `gala`, `green`
   * `basic_type` = `melon`, `sub_types` may be `watermelon`, `honeydew`, `cantaloupe`
   * `basic_type` = `blueberry`
   * `basic_type` = `blackberry`
   * `basic_type` = `strawberry`
   * `basic_type` = `herb`, `sub_types` may be `basil`, `parsley`, `cilantro`, `thyme`
  * fruit cocktail in juice/syrup is `basic_type` = `fruit`, `sub_types` = `["mixed"]`, `processing` = `in juice`, `packaging` = `canned`
  * broccoli normandy is a mixture of broccoli, cauliflower, and carrots, so `basic_type` = `vegetable`, `sub_types` = `["blend"]`
  * haricot vert is green beans
  * winter moon blend is a mixture of butternut squash, potato, carrot, and beets
  * California vegetable blend is a mixture of broccoli, cauliflower, and carrots
  * red rose potato is `basic_type` = `potato`, `sub_types` = `["red"]`
  * cheddar cauliflower is `basic_type` = `cauliflower`, `sub_types` = `["yellow"]`
  * green onions or scallions are `basic_type` = `onion`, `sub_types` = `["green"]`

Seafood & Fish (farm raised or wild):
  * `basic_type`
    * swai, basa are `pangasius`
  * `sub_types`
    * multiple species are `mixed`
    * chum calmon, dog salmon are `basic_type` = `salmon`, `sub_types` = `["keta"]`
    * white tuna is `basic_type` = `tuna`, `sub_types` = `["albacore"]`
    * source (for finfish only, not clams, shrimp, etc.) `sub_types` may be `wild` or `farm-raised`
    * leave out origin information (country of origin, imported, etc.)
  * note `meat_cut`
  * note processing
  * note if `cooked`
  * note dietary concern
    * do not include "MSG" or "no MSG"
  * note additives
  * note frozen
  * note packaging
  * note commodity

Collapsed terms and common abbreviations for fish:
  * cooked
    * `O/R` (oven roasted); collapsed term: `cooked`
    * `PRCK`, `PCK`, `FC`, `PRE-COOKED`; collapsed term: `cooked`
  * cut
    * `FIL`, `FILLET`, `FILET`, `FLT` (filet); collapsed term: `cut`
  * frozen
    * `FRO`, `IQF`; collapsed term: `frozen`
  * processing/preserving
    * `BTR, `BTRD`; collapsed term: `battered`
    * `IN-WAT`, `IN WTR`; collapsed term: `in water`
    * `SMK`; collapsed term: `smoked`

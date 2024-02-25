HINTS = """
        - IW means "individually wrapped" so these should have the "ss" tag in packaging        
        """

EXAMPLES = """Prompt: beef patty 2 oz	
            Output: {
            "Food Product Group": "Meat", 
            "Food Product Category": "Beef", 
            "Basic Type": "beef", 
            "Sub-Type 1": None, 
            "Sub-Type 2": None, 
            "Flavor/Cut": None, 
            "Shape": "patty", 
            "Skin": None, 
            "Seed/Bone": None,
            "Processing": None, 
            "Cooked/Cleaned": None, 
            "WG/WGR": None, 
            "Dietary Concern": None, 
            "Additives": None, 
            "Dietary Accommodation": None, 
            "Frozen": None,
            "Packaging": None, 
            "Commodity": None
            }

            Prompt: CHEESE CUP ULTIMATE CHEDDAR
            Output: {
            "Food Product Group": "Milk & Dairy", 
            "Food Product Category": "Cheese", 
            "Basic Type": "sauce", 
            "Sub-Type 1": "cheese", 
            "Sub-Type 2": "cheddar", 
            "Flavor/Cut": None, 
            "Shape": "patty", 
            "Skin": None, 
            "Seed/Bone": None,
            "Processing": None, 
            "Cooked/Cleaned": None, 
            "WG/WGR": None, 
            "Dietary Concern": None, 
            "Additives": None, 
            "Dietary Accommodation": None, 
            "Frozen": None,
            "Packaging": "ss", 
            "Commodity": None
            }
            """

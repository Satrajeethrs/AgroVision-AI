"""Input validation functions for the crop recommendation system."""

def validate_input_ranges(data):
    """
    Validate input ranges for all parameters.
    Returns (is_valid, error_message)
    """
    validation_rules = {
        'N': {'min': 0, 'max': 200, 'name': 'Nitrogen'},
        'P': {'min': 5, 'max': 150, 'name': 'Phosphorus'},
        'K': {'min': 5, 'max': 200, 'name': 'Potassium'},
        'temperature': {'min': 15, 'max': 45, 'name': 'Temperature'},
        'humidity': {'min': 30, 'max': 100, 'name': 'Humidity'},
        'ph': {'min': 3, 'max': 10, 'name': 'pH'},
        'rainfall': {'min': 100, 'max': 3000, 'name': 'Rainfall'}
    }
    
    for field, rules in validation_rules.items():
        if field not in data:
            return False, f"Missing required field: {rules['name']}"
        
        try:
            value = float(data[field])
            if value < rules['min'] or value > rules['max']:
                return False, f"{rules['name']} must be between {rules['min']} and {rules['max']}"
        except ValueError:
            return False, f"Invalid value for {rules['name']}"
    
    return True, None
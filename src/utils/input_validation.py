"""Input validation functions for the crop recommendation system."""

from src.utils.translation import t

def validate_input_ranges(data, lang='en'):
    """
    Validate input ranges for all parameters.
    Returns (is_valid, error_message)
    """
    validation_rules = {
        'N': {'min': 0, 'max': 200, 'key': 'field.nitrogen'},
        'P': {'min': 5, 'max': 150, 'key': 'field.phosphorus'},
        'K': {'min': 5, 'max': 200, 'key': 'field.potassium'},
        'temperature': {'min': 15, 'max': 45, 'key': 'field.temperature'},
        'humidity': {'min': 30, 'max': 100, 'key': 'field.humidity'},
        'ph': {'min': 3, 'max': 10, 'key': 'field.ph'},
        'rainfall': {'min': 100, 'max': 3000, 'key': 'field.rainfall'}
    }
    
    for field, rules in validation_rules.items():
        field_name = t(rules['key'], lang)
        
        if field not in data:
            error_msg = t('error.missing_field', lang)
            return False, f"{error_msg}: {field_name}"
        
        try:
            value = float(data[field])
            if value < rules['min'] or value > rules['max']:
                error_msg = t('error.out_of_range', lang)
                return False, f"{field_name} {error_msg} {rules['min']} - {rules['max']}"
        except ValueError:
            error_msg = t('error.invalid_value', lang)
            return False, f"{error_msg}: {field_name}"
    
    return True, None
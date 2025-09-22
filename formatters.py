"""
Formatting utilities for agricultural recommendations
"""

def format_crop_recommendation(advice_sections):
    """Format crop recommendation section"""
    prediction = advice_sections.get('prediction', '')
    inputs = advice_sections.get('inputs', {})
    
    return {
        'title': '🌱 Crop Recommendation Analysis',
        'content': f"""
            <div class="recommendation-section">
                <div class="primary-recommendation">
                    <h4>Primary Crop Recommendation: {prediction}</h4>
                    <div class="suitability-analysis">
                        <h5>Suitability Analysis:</h5>
                        <ul>
                            <li>Soil pH: {inputs.get('ph', 0):.1f} - {'Optimal' if 6.0 <= inputs.get('ph', 0) <= 7.5 else 'Needs adjustment'}</li>
                            <li>Temperature: {inputs.get('temperature', 0)}°C - {'Suitable' if 20 <= inputs.get('temperature', 0) <= 35 else 'Monitor closely'}</li>
                            <li>Rainfall: {inputs.get('rainfall', 0)}mm - {'Adequate' if inputs.get('rainfall', 0) > 800 else 'Irrigation needed'}</li>
                            <li>Humidity: {inputs.get('humidity', 0)}% - {'Good' if 50 <= inputs.get('humidity', 0) <= 80 else 'May affect growth'}</li>
                        </ul>
                    </div>
                </div>
            </div>
        """
    }

def format_fertilizer_recommendation(advice_sections):
    """Format fertilizer recommendation section"""
    inputs = advice_sections.get('inputs', {})
    
    return {
        'title': '🧪 Smart Fertilizer Recommendation',
        'content': f"""
            <div class="fertilizer-section">
                <div class="nutrient-analysis">
                    <h4>Nutrient Analysis</h4>
                    <div class="nutrient-levels">
                        <div class="nutrient">
                            <h5>Nitrogen (N)</h5>
                            <p>{inputs.get('N', 0)} kg/ha</p>
                            <span class="badge bg-{'success' if inputs.get('N', 0) > 40 else 'warning'}">
                                {'Adequate' if inputs.get('N', 0) > 40 else 'Low'}
                            </span>
                        </div>
                        <div class="nutrient">
                            <h5>Phosphorus (P)</h5>
                            <p>{inputs.get('P', 0)} kg/ha</p>
                            <span class="badge bg-{'success' if inputs.get('P', 0) > 30 else 'warning'}">
                                {'Good' if inputs.get('P', 0) > 30 else 'Low'}
                            </span>
                        </div>
                        <div class="nutrient">
                            <h5>Potassium (K)</h5>
                            <p>{inputs.get('K', 0)} kg/ha</p>
                            <span class="badge bg-{'success' if inputs.get('K', 0) > 35 else 'warning'}">
                                {'Sufficient' if inputs.get('K', 0) > 35 else 'Low'}
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        """
    }

def format_soil_health(advice_sections):
    """Format soil health section"""
    inputs = advice_sections.get('inputs', {})
    soc = inputs.get('SOC', 0)
    
    return {
        'title': '🌿 Soil Health Management',
        'content': f"""
            <div class="soil-health-section">
                <div class="organic-content">
                    <h4>Soil Organic Carbon (SOC)</h4>
                    <p>{soc:.2f}% - <span class="badge bg-{'success' if soc > 0.5 else 'warning'}">
                        {'Healthy' if soc > 0.5 else 'Needs improvement'}
                    </span></p>
                </div>
                <div class="recommendations mt-4">
                    <h4>Recommendations</h4>
                    <ul>
                        {'<li>Add organic matter through crop residue</li>' if soc < 0.5 else ''}
                        {'<li>Consider green manuring</li>' if soc < 0.4 else ''}
                        {'<li>Maintain current practices</li>' if soc >= 0.5 else ''}
                    </ul>
                </div>
            </div>
        """
    }

def format_disease_analysis(disease_info):
    """Format disease analysis section"""
    if not disease_info:
        return {
            'title': '🔍 Disease Analysis',
            'content': "<p>No disease analysis available.</p>"
        }
        
    return {
        'title': '🔍 Disease Analysis',
        'content': f"""
            <div class="disease-analysis-section">
                <div class="disease-details">
                    <h4>Detected Condition: {disease_info['disease']}</h4>
                    <p><strong>Confidence:</strong> {disease_info['confidence']:.1%}</p>
                    <p><strong>Severity:</strong> <span class="badge bg-{'danger' if disease_info['severity'] == 'High' else 'warning' if disease_info['severity'] == 'Medium' else 'success'}">
                        {disease_info['severity']}
                    </span></p>
                </div>
                <div class="treatment mt-4">
                    <h4>Treatment Recommendations</h4>
                    <p>{disease_info['treatment']}</p>
                </div>
            </div>
        """
    }

def format_results(advice_sections, disease_info=None):
    """Main entry point for formatting all sections"""
    formatted = {
        'executive_summary': advice_sections['executive_summary'],
        'crop_recommendation': format_crop_recommendation(advice_sections),
        'fertilizer_recommendation': format_fertilizer_recommendation(advice_sections),
        'soil_health': format_soil_health(advice_sections)
    }
    
    if disease_info:
        formatted['disease_analysis'] = format_disease_analysis(disease_info)
        
    return formatted
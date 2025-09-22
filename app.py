import os
import pickle
import numpy as np
import pandas as pd
import json
import random
from flask import Flask, render_template, request, jsonify, session, redirect, send_from_directory
from input_validation import validate_input_ranges
from formatters import format_results
from llm_validator import generate_recommendation_text, validate_recommendations
from llm_validator import generate_alternative_recommendation
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import tempfile
from flask import jsonify
from markupsafe import escape as jinja_escape

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session handling

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and scaler
model = None
scaler = None

def load_model_and_scaler():
    """Load the trained model and scaler"""
    global model, scaler
    try:
        # Load the trained model
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully!")

        # Load the scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully!")
        return True
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please run train_model.py first to generate model.pkl and scaler.pkl")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if not request.method == 'POST':
        return redirect('/')

    try:
        form_data = request.form.to_dict()
        logger.info(f"Received form data: {form_data}")

        # Handle plant disease image upload
        disease_pred = None
        disease_conf = None
        disease_solution = None
        if 'disease_image' in request.files and request.files['disease_image'].filename:
            file = request.files['disease_image']
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                file.save(tmp.name)
                img_path = tmp.name
            # Load DL model and class indices
            try:
                dl_model = load_model('plant_disease_model_final.h5')
                class_indices = np.load('plant_disease_class_indices.npy', allow_pickle=True).item()
                idx_to_class = {v: k for k, v in class_indices.items()}
                img = keras_image.load_img(img_path, target_size=(128, 128))
                x = keras_image.img_to_array(img) / 255.0
                x = np.expand_dims(x, axis=0)
                preds = dl_model.predict(x)
                pred_idx = np.argmax(preds, axis=1)[0]
                disease_pred = idx_to_class[pred_idx]
                disease_conf = float(np.max(preds))
                # Simple solution mapping (can be replaced with LLM/API)
                disease_solutions = {
                    'Pepper__bell___Bacterial_spot': 'Remove affected leaves, apply copper-based fungicide.',
                    'Pepper__bell___healthy': 'No disease detected. Maintain good practices.',
                    'Potato___Early_blight': 'Use certified seed, apply fungicide.',
                    'Potato___healthy': 'No disease detected. Maintain good practices.',
                    'Potato___Late_blight': 'Remove infected plants, use resistant varieties.',
                    'Tomato__Target_Spot': 'Apply fungicide, avoid overhead watering.',
                    'Tomato__Tomato_mosaic_virus': 'Remove infected plants, control aphids.',
                    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Control whiteflies, remove infected plants.',
                    'Tomato_Bacterial_spot': 'Use copper sprays, avoid working with wet plants.',
                    'Tomato_Early_blight': 'Apply fungicide, rotate crops.',
                    'Tomato_healthy': 'No disease detected. Maintain good practices.',
                    'Tomato_Late_blight': 'Remove infected leaves, use fungicide.',
                    'Tomato_Leaf_Mold': 'Increase air circulation, use fungicide.',
                    'Tomato_Septoria_leaf_spot': 'Remove affected leaves, apply fungicide.',
                    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Use miticide, increase humidity.'
                }
                disease_solution = disease_solutions.get(disease_pred, 'Consult an expert for treatment advice.')
            except Exception as e:
                disease_pred = 'Error in prediction'
                disease_conf = 0.0
                disease_solution = 'Could not process image.'
        
        # Validate input ranges
        is_valid, validation_error = validate_input_ranges(form_data)
        if not is_valid:
            logger.error(f"Validation error: {validation_error}")
            return render_template('index.html', error_message=validation_error)
        
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Check if model is loaded
        if model is None or scaler is None:
            logger.error("Model or scaler not loaded. Attempting to load...")
            if not load_model_and_scaler():
                return render_template('index.html', 
                    error_message="Error: Model not available. Please contact support.")
        
        try:
            input_features = [float(form_data[name]) for name in feature_names]
            input_df = pd.DataFrame([input_features], columns=feature_names)
            logger.info(f"Processing input features: {input_features}")
            
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]
            probabilities = model.predict_proba(scaled_input)[0]
            confidence = max(probabilities)
            
            logger.info(f"Prediction: {prediction}, Confidence: {confidence:.2%}")
        except Exception as e:
            return render_template('index.html', 
                error_message="Error processing input data. Please check your values.")
        
        # Create structured data for advice generation
        structured_data = {
            'inputs': {name: float(form_data[name]) for name in feature_names},
            'prediction': prediction,
            'confidence': confidence
        }
        
        # Handle disease detection (image-based)
        disease_info = None
        if disease_pred:
            disease_info = {
                'disease': disease_pred,
                'confidence': disease_conf,
                'severity': 'N/A',
                'symptoms': [],
                'treatment': disease_solution
            }
        elif 'disease_symptoms' in form_data and form_data['disease_symptoms'].strip():
            try:
                disease_info = simulate_disease_detection(form_data['disease_symptoms'], prediction)
            except Exception as e:
                return render_template('index.html', 
                    error_message="Error in disease detection. Please try again.")
        
        # Generate comprehensive advice
        try:
            advice_sections = get_comprehensive_advice(structured_data, disease_info)
            # Generate a short AI narrative to include on the results page. This will use
            # the configured LLM provider or the safe stub when no provider is set.
            try:
                ai_out = generate_recommendation_text([
                    {"id": "recommended_crop", "text": f"Recommended crop: {prediction}"}
                ], {"inputs": structured_data['inputs'], "prediction": prediction})
                if isinstance(ai_out, dict):
                    ai_text = ai_out.get('text', '')
                    ai_meta = {'provider': ai_out.get('provider'), 'raw': ai_out.get('raw')}
                else:
                    ai_text = str(ai_out)
                    ai_meta = {'provider': None, 'raw': None}
            except Exception:
                ai_text = ""
                ai_meta = {'provider': None, 'raw': None}

            # Sanitize AI output to avoid untrusted HTML injection; preserve line breaks
            try:
                safe_text = jinja_escape(ai_text)
                safe_text = str(safe_text).replace('\n', '<br>')
            except Exception:
                safe_text = ''

            advice_sections['ai_narrative'] = {
                'title': '🧾 AI Narrative',
                'content': f"<div class=\"ai-narrative\">{safe_text}</div>",
                'meta': ai_meta
            }
            # Generate a hybrid LLM alternative recommendation (not replacing model prediction)
            try:
                alt = generate_alternative_recommendation(structured_data['inputs'])
                alt_crop = alt.get('crop')
                alt_rationale = alt.get('rationale', '')
                alt_meta = {'provider': alt.get('provider'), 'raw': alt.get('raw')}
            except Exception:
                alt_crop = None
                alt_rationale = ''
                alt_meta = {'provider': None, 'raw': None}

            # Sanitize and store
            try:
                safe_rationale = jinja_escape(alt_rationale).replace('\n', '<br>')
            except Exception:
                safe_rationale = ''

            advice_sections['ai_alternative'] = {
                'title': '🤖 LLM Alternative Recommendation',
                'content': f"<div class=\"ai-alternative\"><strong>Alternative:</strong> {alt_crop or 'N/A'}<br>{safe_rationale}</div>",
                'meta': alt_meta
            }
            # expose raw provider info in session for optional debug display
            session['ai_provider_meta'] = ai_meta
            session['advice_sections'] = advice_sections
            session['disease_info'] = disease_info
            return redirect('/results')
        except Exception as e:
            return render_template('index.html', 
                error_message="Error generating advice. Please try again.")
            
    except Exception as e:
        return render_template('index.html', 
            error_message="An unexpected error occurred. Please try again.")

@app.route('/results')
def results():
    # Get analysis results from session
    advice_sections = session.get('advice_sections')
    disease_info = session.get('disease_info')
    
    if not advice_sections:
        return redirect('/')
    
    logger.info(f"Rendering results with sections: {list(advice_sections.keys())}")
    
    return render_template('results.html', 
                         advice_sections=advice_sections,
                         disease_info=disease_info)

def simulate_disease_detection(image_description, crop_type):
    """Simulate disease detection based on symptoms"""
    disease_database = {
        'Rice': [
            {'name': 'Bacterial Leaf Blight', 'symptoms': ['yellow lesions', 'leaf wilting'], 
             'treatment': 'Apply copper-based bactericide', 'severity': 'Medium'},
            {'name': 'Blast Disease', 'symptoms': ['leaf spots', 'neck rot'], 
             'treatment': 'Use fungicide treatment', 'severity': 'High'}
        ],
        'Default': [
            {'name': 'General Disease', 'symptoms': ['wilting', 'spots'], 
             'treatment': 'Consult local agricultural expert', 'severity': 'Medium'}
        ]
    }
    
    diseases = disease_database.get(crop_type, disease_database['Default'])
    disease = random.choice(diseases)
    return {
        'disease': disease['name'],
        'confidence': random.uniform(0.6, 0.9),
        'severity': disease['severity'],
        'symptoms': disease['symptoms'],
        'treatment': disease['treatment']
    }

def get_comprehensive_advice(data, disease_info=None):
    """Generate comprehensive agricultural advice"""
    inputs = data['inputs']
    prediction = data['prediction']
    confidence = data.get('confidence', 0.5)
    
    # Load fertilizer recommendations
    fertilizer_df = pd.read_csv('fertilizer.csv')
    crop_fertilizer = fertilizer_df[fertilizer_df['Crop'].str.lower() == prediction.lower()].iloc[0] if len(fertilizer_df[fertilizer_df['Crop'].str.lower() == prediction.lower()]) > 0 else None
    
    # Calculate status indicators
    n_status = "Low" if inputs['N'] < 50 else "Medium" if inputs['N'] < 100 else "High"
    p_status = "Low" if inputs['P'] < 30 else "Medium" if inputs['P'] < 60 else "High"
    k_status = "Low" if inputs['K'] < 30 else "Medium" if inputs['K'] < 60 else "High"
    ph_status = "Acidic" if inputs['ph'] < 6.0 else "Alkaline" if inputs['ph'] > 7.5 else "Optimal"
    rainfall_status = "Low" if inputs['rainfall'] < 500 else "High" if inputs['rainfall'] > 2000 else "Optimal"
    
    # Calculate nutrient deficiencies
    n_status = "Low" if inputs['N'] < 50 else "Medium" if inputs['N'] < 100 else "High"
    p_status = "Low" if inputs['P'] < 30 else "Medium" if inputs['P'] < 60 else "High"
    k_status = "Low" if inputs['K'] < 30 else "Medium" if inputs['K'] < 60 else "High"
    
    advice_sections = {
        'executive_summary': {
            'title': '📋 Executive Summary',
            'content': f"""
                <div class="executive-summary">
                    <div class="summary-header">
                        <h3>Crop Recommendation Summary</h3>
                        <div class="recommendation-badge">
                            <strong>Recommended Crop:</strong> 
                            <span class="badge bg-success">{prediction}</span>
                            <span class="confidence-badge">Confidence: {confidence:.1%}</span>
                        </div>
                    </div>
                    
                    <div class="summary-grid">
                        <div class="summary-card soil-nutrients">
                            <h4>Soil Nutrient Status</h4>
                            <div class="nutrient-indicators">
                                <div class="indicator {n_status.lower()}">
                                    <span>N: {inputs['N']} kg/ha</span>
                                    <span class="badge">{n_status}</span>
                                </div>
                                <div class="indicator {p_status.lower()}">
                                    <span>P: {inputs['P']} kg/ha</span>
                                    <span class="badge">{p_status}</span>
                                </div>
                                <div class="indicator {k_status.lower()}">
                                    <span>K: {inputs['K']} kg/ha</span>
                                    <span class="badge">{k_status}</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="summary-card environmental">
                            <h4>Environmental Conditions</h4>
                            <div class="env-conditions">
                                <div class="condition">
                                    <span>Temperature</span>
                                    <span>{inputs['temperature']}°C</span>
                                </div>
                                <div class="condition">
                                    <span>Humidity</span>
                                    <span>{inputs['humidity']}%</span>
                                </div>
                                <div class="condition">
                                    <span>Rainfall</span>
                                    <span class="badge {rainfall_status.lower()}">{inputs['rainfall']} mm</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="summary-card soil-conditions">
                            <h4>Soil Conditions</h4>
                            <div class="soil-indicators">
                                <div class="indicator">
                                    <span>Soil pH</span>
                                    <span class="badge {ph_status.lower()}">{inputs['ph']} ({ph_status})</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="key-recommendations">
                        <h4>Key Recommendations</h4>
                        <ul>
                            {f'<li>pH Management: {get_ph_management_advice(inputs["ph"]).strip("</p>").strip("<p>")}</li>' if ph_status != "Optimal" else ''}
                            {f'<li>Rainfall: {get_rainfall_advice(inputs["rainfall"])}</li>' if rainfall_status != "Optimal" else ''}
                            {"<li>Consider adding nitrogen fertilizer</li>" if n_status == "Low" else ""}
                            {"<li>Consider adding phosphorus</li>" if p_status == "Low" else ""}
                            {"<li>Consider adding potassium</li>" if k_status == "Low" else ""}
                        </ul>
                    </div>
                </div>
            """
        },
        'crop_recommendation': {
            'title': '🌱 Crop Recommendation Analysis',
            'content': f"""
                <div class="crop-analysis">
                    <div class="recommendation-header">
                        <h3>Primary Crop Recommendation: {prediction}</h3>
                        <span class="confidence-pill">AI Confidence: {confidence:.1%}</span>
                    </div>

                    <div class="analysis-section">
                        <h4>Detailed Suitability Analysis</h4>
                        <div class="suitability-grid">
                            <div class="factor-card {get_ph_status_class(inputs['ph'])}">
                                <h5>Soil pH</h5>
                                <div class="value">{inputs['ph']:.1f}</div>
                                <div class="status">{get_ph_advice(inputs['ph'])}</div>
                                <div class="recommendation">
                                    {get_ph_management_advice(inputs['ph']).strip("<p>").strip("</p>")}
                                </div>
                            </div>

                            <div class="factor-card {get_temperature_status_class(inputs['temperature'])}">
                                <h5>Temperature</h5>
                                <div class="value">{inputs['temperature']}°C</div>
                                <div class="status">{get_temperature_advice(inputs['temperature'])}</div>
                                <div class="recommendation">
                                    {get_temperature_management(inputs['temperature'])}
                                </div>
                            </div>

                            <div class="factor-card {get_rainfall_status_class(inputs['rainfall'])}">
                                <h5>Rainfall</h5>
                                <div class="value">{inputs['rainfall']} mm</div>
                                <div class="status">{get_rainfall_advice(inputs['rainfall'])}</div>
                                <div class="recommendation">
                                    {get_rainfall_management(inputs['rainfall'])}
                                </div>
                            </div>

                            <div class="factor-card {get_humidity_status_class(inputs['humidity'])}">
                                <h5>Humidity</h5>
                                <div class="value">{inputs['humidity']}%</div>
                                <div class="status">{get_humidity_advice(inputs['humidity'])}</div>
                                <div class="recommendation">
                                    {get_humidity_management(inputs['humidity'])}
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="crop-requirements">
                        <h4>Optimal Growing Conditions for {prediction}</h4>
                        <div class="requirements-table">
                            <table class="table table-bordered">
                                <thead>
                                    <tr>
                                        <th>Factor</th>
                                        <th>Optimal Range</th>
                                        <th>Current Value</th>
                                        <th>Status</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>Soil pH</td>
                                        <td>6.0 - 7.5</td>
                                        <td>{inputs['ph']:.1f}</td>
                                        <td><span class="badge {ph_status.lower()}">{ph_status}</span></td>
                                    </tr>
                                    <tr>
                                        <td>Temperature</td>
                                        <td>20°C - 30°C</td>
                                        <td>{inputs['temperature']}°C</td>
                                        <td><span class="badge {get_temperature_status_class(inputs['temperature'])}">{get_temperature_status(inputs['temperature'])}</span></td>
                                    </tr>
                                    <tr>
                                        <td>Rainfall</td>
                                        <td>500 - 2000 mm</td>
                                        <td>{inputs['rainfall']} mm</td>
                                        <td><span class="badge {rainfall_status.lower()}">{rainfall_status}</span></td>
                                    </tr>
                                    <tr>
                                        <td>Humidity</td>
                                        <td>30% - 90%</td>
                                        <td>{inputs['humidity']}%</td>
                                        <td><span class="badge {get_humidity_status_class(inputs['humidity'])}">{get_humidity_status(inputs['humidity'])}</span></td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            """
        },
        'fertilizer_recommendation': {
            'title': '🧪 Smart Fertilizer Recommendation',
            'content': f"""
                <div class="fertilizer-analysis">
                    <div class="current-status">
                        <h3>Current Nutrient Levels</h3>
                        <div class="nutrient-grid">
                            <div class="nutrient-card {n_status.lower()}">
                                <div class="nutrient-header">
                                    <h4>Nitrogen (N)</h4>
                                    <span class="status-badge {n_status.lower()}">{n_status}</span>
                                </div>
                                <div class="nutrient-value">{inputs['N']} kg/ha</div>
                                <div class="optimal-range">Optimal: 50-100 kg/ha</div>
                                <div class="recommendation">
                                    {get_detailed_nutrient_advice('N', inputs['N'], n_status)}
                                </div>
                            </div>
                            <div class="nutrient-card {p_status.lower()}">
                                <div class="nutrient-header">
                                    <h4>Phosphorus (P)</h4>
                                    <span class="status-badge {p_status.lower()}">{p_status}</span>
                                </div>
                                <div class="nutrient-value">{inputs['P']} kg/ha</div>
                                <div class="optimal-range">Optimal: 30-60 kg/ha</div>
                                <div class="recommendation">
                                    {get_detailed_nutrient_advice('P', inputs['P'], p_status)}
                                </div>
                            </div>
                            <div class="nutrient-card {k_status.lower()}">
                                <div class="nutrient-header">
                                    <h4>Potassium (K)</h4>
                                    <span class="status-badge {k_status.lower()}">{k_status}</span>
                                </div>
                                <div class="nutrient-value">{inputs['K']} kg/ha</div>
                                <div class="optimal-range">Optimal: 30-60 kg/ha</div>
                                <div class="recommendation">
                                    {get_detailed_nutrient_advice('K', inputs['K'], k_status)}
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="fertilizer-recommendations">
                        <h3>Recommended Actions</h3>
                        <div class="action-items">
                            <ul class="recommendation-list">
                                {generate_fertilizer_action_items(n_status, p_status, k_status, prediction)}
                            </ul>
                        </div>
                    </div>

                    <div class="application-guidelines">
                        <h3>Application Guidelines</h3>
                        <div class="guidelines-box">
                            <h4>Best Practices</h4>
                            <ul>
                                <li>Apply fertilizers in the early morning or late evening</li>
                                <li>Ensure soil is moist before application</li>
                                <li>Incorporate fertilizers into the soil rather than surface application</li>
                                <li>Split the application into multiple doses for better absorption</li>
                            </ul>
                            
                            <h4>Timing</h4>
                            <p>For {prediction}:</p>
                            {get_crop_specific_timing(prediction)}
                        </div>
                    </div>
                </div>
            """
        },
        'soil_health': {
            'title': '🌿 Soil Health Management',
            'content': f"""
                <div class="soil-health-box">
                    <div class="ph-management">
                        <h3>Soil pH Management</h3>
                        {get_ph_management_advice(inputs['ph'])}
                        <div class="recommendations mt-4">
                            <h4>pH Improvement Guidelines</h4>
                            <ul>
                                {'<li>Add lime to increase pH (reduce acidity)</li>' if inputs['ph'] < 5.5 else ''}
                                {'<li>Add sulfur to decrease pH (reduce alkalinity)</li>' if inputs['ph'] > 7.5 else ''}
                                {'<li>Current pH is optimal - maintain existing practices</li>' if 5.5 <= inputs['ph'] <= 7.5 else ''}
                            </ul>
                        </div>
                    </div>
                    <div class="soil-management-practices mt-4">
                        <h3>General Soil Management Practices</h3>
                        <ul>
                            <li>Practice crop rotation to maintain soil health</li>
                            <li>Add organic matter through composting</li>
                            <li>Implement proper drainage systems</li>
                            <li>Monitor soil moisture regularly</li>
                        </ul>
                    </div>
                </div>
            """
        }
    }
    
    if disease_info:
        advice_sections['disease_analysis'] = {
            'title': '🔍 Disease Analysis',
            'content': f"""
                <div class="disease-analysis-box">
                    <h3>Disease Detection Results</h3>
                    <div class="disease-info">
                        <p><strong>Detected Disease:</strong> {disease_info['disease']}</p>
                        <p><strong>Confidence:</strong> {disease_info['confidence']:.1%}</p>
                        <p><strong>Severity:</strong> <span class="badge bg-warning">{disease_info['severity']}</span></p>
                    </div>
                    <div class="treatment-info">
                        <h4>Treatment Recommendation</h4>
                        <p>{disease_info['treatment']}</p>
                    </div>
                </div>
            """
        }
    
    return advice_sections

def get_ph_advice(ph):
    if ph < 5.5:
        return "Needs adjustment - Too acidic"
    elif ph > 7.5:
        return "Needs adjustment - Too alkaline"
    else:
        return "Optimal range"

def get_temperature_advice(temp):
    if temp < 15:
        return "Monitor closely - Too cold"
    elif temp > 35:
        return "Monitor closely - Too hot"
    else:
        return "Optimal range"

def get_rainfall_advice(rainfall):
    if rainfall < 500:
        return "Irrigation needed"
    elif rainfall > 2000:
        return "Drainage may be required"
    else:
        return "Adequate"

def get_humidity_advice(humidity):
    if humidity < 30:
        return "May affect growth - Too dry"
    elif humidity > 90:
        return "Monitor for diseases - Too humid"
    else:
        return "Optimal range"

def get_nutrient_recommendation(nutrient, current, target):
    if target is None:
        return "<p>No specific target available for this crop</p>"
    
    diff = target - current
    if abs(diff) < 10:
        return "<p>Current levels are appropriate</p>"
    elif diff > 0:
        return f"<p>Increase by approximately {diff:.0f} kg/ha</p>"
    else:
        return f"<p>Reduce by approximately {abs(diff):.0f} kg/ha</p>"



def get_detailed_nutrient_advice(nutrient, value, status):
    if nutrient == 'N':
        if status == 'Low':
            return """
                <ul>
                    <li>Apply nitrogen-rich fertilizers like urea or ammonium sulfate</li>
                    <li>Consider adding organic matter like compost</li>
                    <li>Add nitrogen-fixing cover crops</li>
                </ul>
            """
        elif status == 'High':
            return """
                <ul>
                    <li>Reduce nitrogen fertilizer application</li>
                    <li>Plant heavy nitrogen-feeding crops</li>
                    <li>Add carbon-rich organic matter</li>
                </ul>
            """
        else:
            return "<p>Current nitrogen levels are optimal. Maintain current practices.</p>"
    
    elif nutrient == 'P':
        if status == 'Low':
            return """
                <ul>
                    <li>Apply phosphate fertilizers</li>
                    <li>Add bone meal or rock phosphate</li>
                    <li>Maintain soil pH between 6.0-7.0</li>
                </ul>
            """
        elif status == 'High':
            return """
                <ul>
                    <li>Avoid phosphorus fertilizers</li>
                    <li>Monitor water quality</li>
                    <li>Consider phosphorus-feeding crops</li>
                </ul>
            """
        else:
            return "<p>Current phosphorus levels are optimal. Maintain current practices.</p>"
    
    else:  # Potassium (K)
        if status == 'Low':
            return """
                <ul>
                    <li>Apply potassium-rich fertilizers</li>
                    <li>Add wood ash or kelp meal</li>
                    <li>Maintain proper soil drainage</li>
                </ul>
            """
        elif status == 'High':
            return """
                <ul>
                    <li>Avoid potassium fertilizers</li>
                    <li>Improve soil drainage</li>
                    <li>Plant potassium-feeding crops</li>
                </ul>
            """
        else:
            return "<p>Current potassium levels are optimal. Maintain current practices.</p>"

def generate_fertilizer_action_items(n_status, p_status, k_status, crop):
    actions = []
    
    # Load crop-specific requirements
    fertilizer_df = pd.read_csv('fertilizer.csv')
    crop_req = fertilizer_df[fertilizer_df['Crop'].str.lower() == crop.lower()].iloc[0] if len(fertilizer_df[fertilizer_df['Crop'].str.lower() == crop.lower()]) > 0 else None
    
    # Add nutrient-specific recommendations
    if n_status == 'Low':
        actions.append("<li><strong>Nitrogen (N):</strong> Apply nitrogen-rich fertilizer in split doses. Consider urea or ammonium sulfate.</li>")
    elif n_status == 'High':
        actions.append("<li><strong>Nitrogen (N):</strong> Reduce nitrogen application. Consider planting nitrogen-feeding crops.</li>")
    
    if p_status == 'Low':
        actions.append("<li><strong>Phosphorus (P):</strong> Add phosphate fertilizers or organic alternatives like bone meal.</li>")
    elif p_status == 'High':
        actions.append("<li><strong>Phosphorus (P):</strong> Avoid phosphorus-rich fertilizers for the next season.</li>")
    
    if k_status == 'Low':
        actions.append("<li><strong>Potassium (K):</strong> Apply potassium-rich fertilizers like potassium sulfate.</li>")
    elif k_status == 'High':
        actions.append("<li><strong>Potassium (K):</strong> Reduce potassium fertilizer application.</li>")
    
    # Add crop-specific advice
    if crop_req is not None:
        actions.append(f"<li><strong>Crop-Specific:</strong> Optimal N-P-K ratio for {crop}: {crop_req['N']}-{crop_req['P']}-{crop_req['K']}</li>")
    
    # Add general recommendations
    actions.append("<li><strong>Soil Testing:</strong> Regular soil testing every 2-3 years is recommended.</li>")
    actions.append("<li><strong>pH Management:</strong> Maintain proper pH levels for optimal nutrient absorption.</li>")
    
    return "\n".join(actions)

def get_crop_specific_timing(crop):
    # Define timing recommendations for different crops
    timing_recommendations = {
        'rice': """
            <ul>
                <li>Base fertilizer: Apply during land preparation</li>
                <li>Top dressing: 25-30 days after transplanting</li>
                <li>Second top dressing: During panicle initiation</li>
            </ul>
        """,
        'wheat': """
            <ul>
                <li>Base fertilizer: At sowing time</li>
                <li>First top dressing: 20-25 days after sowing</li>
                <li>Second top dressing: 45-50 days after sowing</li>
            </ul>
        """,
        'maize': """
            <ul>
                <li>Base fertilizer: At sowing time</li>
                <li>First top dressing: 25-30 days after emergence</li>
                <li>Second top dressing: At tasseling stage</li>
            </ul>
        """
    }
    
    # Return crop-specific timing or default recommendations
    return timing_recommendations.get(crop.lower(), """
        <ul>
            <li>Base application: During soil preparation or at planting</li>
            <li>Top dressing: During peak growth period</li>
            <li>Follow local agricultural extension service guidelines</li>
        </ul>
    """)

def get_ph_status_class(ph):
    if ph < 5.5:
        return "warning"
    elif ph > 7.5:
        return "warning"
    return "success"

def get_temperature_status(temp):
    if temp < 15:
        return "Too Cold"
    elif temp > 35:
        return "Too Hot"
    return "Optimal"

def get_temperature_status_class(temp):
    if temp < 15 or temp > 35:
        return "warning"
    return "success"

def get_temperature_management(temp):
    if temp < 15:
        return "Consider greenhouse cultivation or wait for warmer season"
    elif temp > 35:
        return "Provide shade and ensure adequate irrigation"
    return "Maintain current practices"

def get_rainfall_status_class(rainfall):
    if rainfall < 500:
        return "warning"
    elif rainfall > 2000:
        return "warning"
    return "success"

def get_rainfall_management(rainfall):
    if rainfall < 500:
        return "Implement irrigation system and water conservation practices"
    elif rainfall > 2000:
        return "Ensure proper drainage and consider raised beds"
    return "Maintain current practices"

def get_humidity_status(humidity):
    if humidity < 30:
        return "Too Dry"
    elif humidity > 90:
        return "Too Humid"
    return "Optimal"

def get_humidity_status_class(humidity):
    if humidity < 30 or humidity > 90:
        return "warning"
    return "success"

def get_humidity_management(humidity):
    if humidity < 30:
        return "Consider misting systems or humidity trays"
    elif humidity > 90:
        return "Improve ventilation and air circulation"
    return "Maintain current practices"

def get_ph_management_advice(ph):
    if ph < 5.5:
        return "<p>Consider adding lime to increase soil pH</p>"
    elif ph > 7.5:
        return "<p>Consider adding sulfur to decrease soil pH</p>"
    else:
        return "<p>pH levels are optimal - maintain current practices</p>"


def _strip_tags(html: str) -> str:
    """Very small helper to remove HTML tags for passing plain text to validator."""
    if not html:
        return ""
    return re.sub(r'<[^>]*>', '', html)


@app.route('/validate_recs', methods=['POST'])
def validate_recs_route():
    # Validate recommendations on-demand using session `advice_sections`
    advice_sections = session.get('advice_sections')
    if not advice_sections:
        return jsonify({'error': 'No analysis available in session'}), 400

    # Build compact recs list from key sections
    recs = []
    keys = ['crop_recommendation', 'fertilizer_recommendation', 'executive_summary']
    for k in keys:
        sec = advice_sections.get(k, {})
        content = sec.get('content', '')
        text = _strip_tags(content)
        if text:
            recs.append({'id': k, 'text': text})

    data_summary = {'note': 'derived from session'}
    out = validate_recommendations(recs, data_summary)
    return jsonify(out)

if __name__ == '__main__':
    if load_model_and_scaler():
        app.run(debug=True, port=5001)
    else:
        print("Failed to start the application due to missing model files.")
import os
import pickle
import numpy as np
import pandas as pd
import json
import random
from flask import Flask, render_template, request, jsonify, session, redirect, send_from_directory
from dotenv import load_dotenv
from src.utils.input_validation import validate_input_ranges
from src.utils.formatters import format_results
from src.utils.llm_validator import generate_recommendation_text, validate_recommendations
from src.utils.llm_validator import generate_alternative_recommendation
from src.utils.translation import get_translation_service, t, translate, get_supported_languages
from src.utils.chatbot import create_chatbot
import re
import tempfile
from flask import jsonify
from markupsafe import escape as jinja_escape

# Note: TensorFlow imports are lazy-loaded inside functions to speed up startup

# Load environment variables from .env (HF_TOKEN, etc.)
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session handling

# Make translation functions available in templates
@app.context_processor
def inject_translation():
    """Make translation function available in all templates"""
    def translate_ui(key, **kwargs):
        lang = session.get('language', 'en')
        return t(key, lang, **kwargs)
    return dict(t=translate_ui, current_lang=lambda: session.get('language', 'en'))

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and scaler
model = None
scaler = None

# Translation helper function
def get_user_language():
    """Get user's preferred language from session"""
    return session.get('language', 'en')

def translate_text(text, target_lang=None):
    """Translate text to user's language"""
    if target_lang is None:
        target_lang = get_user_language()
    
    # If English or no translation needed, return as-is
    if target_lang == 'en' or not text:
        return text
    
    # Use translation service
    return translate(text, target_lang=target_lang, source_lang='en')

def get_translated_status(key, lang=None):
    """Get translated status text"""
    if lang is None:
        lang = get_user_language()
    return t(f'status.{key.lower()}', lang)

def load_model_and_scaler():
    """Load the ML model and scaler"""
    global model, scaler
    try:
        print("  📂 Loading crop recommendation model...")
        # Load the trained model
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
        print("  ✓ Model loaded")
        
        print("  📂 Loading scaler...")
        # Load the scaler
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Scaler loaded successfully")
        print("  ✓ Scaler loaded")
        return True
    except Exception as e:
        logger.error(f"Error loading model or scaler: {e}")
        print(f"  ✗ Error: {e}")
        return False

@app.route('/')
def index():
    # Set default language if not in session
    if 'language' not in session:
        session['language'] = 'en'
    return render_template('index.html')

@app.route('/set_language', methods=['POST'])
def set_language():
    """Endpoint to set user's preferred language"""
    try:
        data = request.get_json()
        language = data.get('language', 'en')
        
        # Validate language is supported
        supported = get_supported_languages()
        if language in supported:
            session['language'] = language
            logger.info(f"Language set to: {language}")
            return jsonify({'status': 'success', 'language': language})
        else:
            return jsonify({'status': 'error', 'message': 'Unsupported language'}), 400
    except Exception as e:
        logger.error(f"Error setting language: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def api_translate():
    """API endpoint for JavaScript to fetch translations"""
    try:
        data = request.get_json()
        keys = data.get('keys', [])
        lang = session.get('language', 'en')
        
        translations = {}
        for key in keys:
            translations[key] = t(key, lang)
        
        return jsonify({'status': 'success', 'translations': translations, 'lang': lang})
    except Exception as e:
        logger.error(f"Error in translation API: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

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
                # Lazy load TensorFlow to speed up startup
                from tensorflow.keras.models import load_model
                from tensorflow.keras.preprocessing import image as keras_image
                
                dl_model = load_model('models/plant_disease_model_final.h5')
                class_indices = np.load('models/plant_disease_class_indices.npy', allow_pickle=True).item()
                idx_to_class = {v: k for k, v in class_indices.items()}
                img = keras_image.load_img(img_path, target_size=(128, 128))
                x = keras_image.img_to_array(img) / 255.0
                x = np.expand_dims(x, axis=0)
                preds = dl_model.predict(x)
                pred_idx = np.argmax(preds, axis=1)[0]
                disease_pred = idx_to_class[pred_idx]
                disease_conf = float(np.max(preds))
                # Get user language for translation
                lang = get_user_language()
                # Simple solution mapping (translated)
                disease_solutions_keys = {
                    'Pepper__bell___Bacterial_spot': 'disease.pepper_bacterial_spot',
                    'Pepper__bell___healthy': 'disease.pepper_healthy',
                    'Potato___Early_blight': 'disease.potato_early_blight',
                    'Potato___healthy': 'disease.potato_healthy',
                    'Potato___Late_blight': 'disease.potato_late_blight',
                    'Tomato__Target_Spot': 'disease.tomato_target_spot',
                    'Tomato__Tomato_mosaic_virus': 'disease.tomato_mosaic_virus',
                    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'disease.tomato_yellow_leaf_curl',
                    'Tomato_Bacterial_spot': 'disease.tomato_bacterial_spot',
                    'Tomato_Early_blight': 'disease.tomato_early_blight',
                    'Tomato_healthy': 'disease.tomato_healthy',
                    'Tomato_Late_blight': 'disease.tomato_late_blight',
                    'Tomato_Leaf_Mold': 'disease.tomato_leaf_mold',
                    'Tomato_Septoria_leaf_spot': 'disease.tomato_septoria_leaf_spot',
                    'Tomato_Spider_mites_Two_spotted_spider_mite': 'disease.tomato_spider_mites'
                }
                solution_key = disease_solutions_keys.get(disease_pred, 'disease.consult_expert')
                disease_solution = t(solution_key, lang)
            except Exception as e:
                disease_pred = t('error.prediction', lang)
                disease_conf = 0.0
                disease_solution = t('error.image_processing', lang)
        
        # Ensure required numeric fields exist (provide sensible defaults if missing)
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        default_values = {
            'N': 90, 'P': 42, 'K': 43,
            'temperature': 26, 'humidity': 80,
            'ph': 6.5, 'rainfall': 1200
        }
        for k in feature_names:
            if k not in form_data or str(form_data.get(k)).strip() == '':
                form_data[k] = str(default_values[k])

        # Validate input ranges
        lang = get_user_language()
        is_valid, validation_error = validate_input_ranges(form_data, lang)
        if not is_valid:
            logger.error(f"Validation error: {validation_error}")
            return render_template('index.html', error_message=validation_error)
        
        
        # Check if model is loaded
        if model is None or scaler is None:
            logger.error("Model or scaler not loaded. Attempting to load...")
            if not load_model_and_scaler():
                lang = get_user_language()
                return render_template('index.html', 
                    error_message=t('error.model_not_loaded', lang))
        
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
            lang = get_user_language()
            return render_template('index.html', 
                error_message=t('error.processing_input', lang))
        
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
                lang = get_user_language()
                return render_template('index.html', 
                    error_message=t('error.disease_detection', lang))
        
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
                
                # Translate AI text if not in English
                lang = get_user_language()
                if lang != 'en' and ai_text:
                    try:
                        ai_text = translate(ai_text, lang, 'en')
                    except Exception as e:
                        logger.warning(f"Failed to translate AI narrative: {e}")
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
                'title': f'🧾 {t("section.ai_narrative", lang)}',
                'content': f"<div class=\"ai-narrative\">{safe_text}</div>",
                'meta': ai_meta
            }
            # Generate a hybrid LLM alternative recommendation (not replacing model prediction)
            try:
                alt = generate_alternative_recommendation(structured_data['inputs'])
                alt_crop = alt.get('crop')
                alt_rationale = alt.get('rationale', '')
                alt_meta = {'provider': alt.get('provider'), 'raw': alt.get('raw')}
                
                # Translate alternative recommendation if not in English
                if lang != 'en':
                    if alt_crop:
                        try:
                            alt_crop = translate(alt_crop, lang, 'en')
                        except Exception as e:
                            logger.warning(f"Failed to translate alternative crop: {e}")
                    if alt_rationale:
                        try:
                            alt_rationale = translate(alt_rationale, lang, 'en')
                        except Exception as e:
                            logger.warning(f"Failed to translate alternative rationale: {e}")
            except Exception:
                alt_crop = None
                alt_rationale = ''
                alt_meta = {'provider': None, 'raw': None}

            # Sanitize and store
            try:
                safe_rationale = jinja_escape(alt_rationale).replace('\n', '<br>')
            except Exception:
                safe_rationale = ''
            
            alt_label = t('text.alternative', lang)
            advice_sections['ai_alternative'] = {
                'title': f'🤖 {alt_label} {t("section.crop_recommendation", lang)}',
                'content': f"<div class=\"ai-alternative\"><strong>{alt_label}:</strong> {alt_crop or t('text.na', lang)}<br>{safe_rationale}</div>",
                'meta': alt_meta
            }
            # expose raw provider info in session for optional debug display
            session['ai_provider_meta'] = ai_meta
            session['advice_sections'] = advice_sections
            session['disease_info'] = disease_info
            return redirect('/results')
        except Exception as e:
            lang = get_user_language()
            return render_template('index.html', 
                error_message=t('error.generating_advice', lang))
            
    except Exception as e:
        lang = get_user_language()
        return render_template('index.html', 
            error_message=t('error.unexpected', lang))

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
    
    # Get user's language
    lang = get_user_language()
    
    # Translate crop name if not English
    translated_prediction = translate(prediction, lang, 'en') if lang != 'en' else prediction
    
    # Load fertilizer recommendations
    fertilizer_df = pd.read_csv('data/raw/fertilizer.csv')
    crop_fertilizer = fertilizer_df[fertilizer_df['Crop'].str.lower() == prediction.lower()].iloc[0] if len(fertilizer_df[fertilizer_df['Crop'].str.lower() == prediction.lower()]) > 0 else None
    
    # Calculate status indicators (English keys for CSS classes)
    n_status = "Low" if inputs['N'] < 50 else "Medium" if inputs['N'] < 100 else "High"
    p_status = "Low" if inputs['P'] < 30 else "Medium" if inputs['P'] < 60 else "High"
    k_status = "Low" if inputs['K'] < 30 else "Medium" if inputs['K'] < 60 else "High"
    ph_status = "Acidic" if inputs['ph'] < 6.0 else "Alkaline" if inputs['ph'] > 7.5 else "Optimal"
    rainfall_status = "Low" if inputs['rainfall'] < 500 else "High" if inputs['rainfall'] > 2000 else "Optimal"
    
    # Get translated status labels
    n_status_text = t(f'status.{n_status.lower()}', lang)
    p_status_text = t(f'status.{p_status.lower()}', lang)
    k_status_text = t(f'status.{k_status.lower()}', lang)
    ph_status_text = t(f'status.{ph_status.lower()}', lang)
    rainfall_status_text = t(f'status.{rainfall_status.lower()}', lang)
    
    # Translate static labels
    label_exec_summary = t('section.executive_summary', lang)
    label_crop_rec = t('section.crop_recommendation', lang)
    label_fert_rec = t('section.fertilizer_recommendation', lang)
    label_soil_health_mgmt = t('section.soil_health', lang)
    label_crop_rec_summary = t('text.crop_rec_summary', lang)
    label_recommended_crop = t('result.recommended_crop', lang)
    label_confidence = t('result.confidence', lang)
    label_soil_nutrient_status = t('result.soil_nutrient_status', lang)
    label_env_conditions = t('result.environmental_conditions', lang)
    label_soil_conditions = t('result.soil_conditions', lang)
    label_key_recs = t('result.key_recommendations', lang)
    label_temperature = t('ui.temperature', lang)
    label_humidity = t('ui.humidity', lang)
    label_rainfall = t('ui.rainfall', lang)
    label_soil_ph = t('ui.ph', lang)
    label_ph_mgmt = t('advice.ph_management', lang)
    label_add_n = t('advice.consider_nitrogen', lang)
    label_add_p = t('advice.consider_phosphorus', lang)
    label_add_k = t('advice.consider_potassium', lang)
    
    # Additional labels for detailed sections
    label_primary_crop = t('text.primary_crop_rec', lang)
    label_ai_confidence = t('text.ai_confidence', lang)
    label_detailed_analysis = t('text.detailed_analysis', lang)
    label_soil_ph_label = t('ui.ph', lang)
    label_temp_label = t('ui.temperature', lang)
    label_rainfall_label = t('ui.rainfall', lang)
    label_humidity_label = t('ui.humidity', lang)
    label_optimal_conditions = t('text.optimal_conditions', lang)
    label_factor = t('text.factor', lang)
    label_optimal_range = t('text.optimal_range', lang)
    label_current_value = t('text.current_value', lang)
    label_status = t('text.status', lang)
    label_current_nutrient = t('text.current_nutrient_levels', lang)
    label_nitrogen = t('ui.nitrogen', lang)
    label_phosphorus = t('ui.phosphorus', lang)
    label_potassium = t('ui.potassium', lang)
    label_optimal = t('text.optimal', lang)
    label_recommended_actions = t('text.recommended_actions', lang)
    label_application_guidelines = t('text.application_guidelines', lang)
    label_best_practices = t('text.best_practices', lang)
    label_timing = t('text.timing', lang)
    label_for = t('text.for', lang)
    label_soil_ph_mgmt = t('text.soil_ph_management', lang)
    label_ph_improvement = t('text.ph_improvement', lang)
    label_general_soil_mgmt = t('text.general_soil_mgmt', lang)
    label_maintain_practices = t('text.levels_optimal', lang)
    
    advice_sections = {
        'executive_summary': {
            'title': f'📋 {label_exec_summary}',
            'content': f"""
                <div class="executive-summary">
                    <div class="summary-header">
                        <h3>{label_crop_rec_summary}</h3>
                        <div class="recommendation-badge">
                            <strong>{label_recommended_crop}:</strong> 
                            <span class="badge bg-success">{translated_prediction}</span>
                            <span class="confidence-badge">{label_confidence}: {confidence:.1%}</span>
                        </div>
                    </div>
                    
                    <div class="summary-grid">
                        <div class="summary-card soil-nutrients">
                            <h4>{label_soil_nutrient_status}</h4>
                            <div class="nutrient-indicators">
                                <div class="indicator {n_status.lower()}">
                                    <span>N: {inputs['N']} {t('ui.unit_kgha', lang)}</span>
                                    <span class="badge">{n_status_text}</span>
                                </div>
                                <div class="indicator {p_status.lower()}">
                                    <span>P: {inputs['P']} {t('ui.unit_kgha', lang)}</span>
                                    <span class="badge">{p_status_text}</span>
                                </div>
                                <div class="indicator {k_status.lower()}">
                                    <span>K: {inputs['K']} {t('ui.unit_kgha', lang)}</span>
                                    <span class="badge">{k_status_text}</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="summary-card environmental">
                            <h4>{label_env_conditions}</h4>
                            <div class="env-conditions">
                                <div class="condition">
                                    <span>{label_temperature}</span>
                                    <span>{inputs['temperature']}{t('ui.unit_celsius', lang)}</span>
                                </div>
                                <div class="condition">
                                    <span>{label_humidity}</span>
                                    <span>{inputs['humidity']}{t('ui.unit_percent', lang)}</span>
                                </div>
                                <div class="condition">
                                    <span>{label_rainfall}</span>
                                    <span class="badge {rainfall_status.lower()}">{inputs['rainfall']} {t('ui.unit_mm', lang)}</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="summary-card soil-conditions">
                            <h4>{label_soil_conditions}</h4>
                            <div class="soil-indicators">
                                <div class="indicator">
                                    <span>{label_soil_ph}</span>
                                    <span class="badge {ph_status.lower()}">{inputs['ph']} ({ph_status_text})</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="key-recommendations">
                        <h4>{label_key_recs}</h4>
                        <ul>
                            {f'<li>{label_ph_mgmt}: {get_ph_management_advice(inputs["ph"], lang).strip("</p>").strip("<p>")}</li>' if ph_status != "Optimal" else ''}
                            {f'<li>{label_rainfall}: {get_rainfall_advice(inputs["rainfall"], lang)}</li>' if rainfall_status != "Optimal" else ''}
                            {f"<li>{label_add_n}</li>" if n_status == "Low" else ""}
                            {f"<li>{label_add_p}</li>" if p_status == "Low" else ""}
                            {f"<li>{label_add_k}</li>" if k_status == "Low" else ""}
                        </ul>
                    </div>
                </div>
            """
        },
        'crop_recommendation': {
            'title': f'🌱 {label_crop_rec}',
            'content': f"""
                <div class="crop-analysis">
                    <div class="recommendation-header">
                        <h3>{label_primary_crop}: {translated_prediction}</h3>
                        <span class="confidence-pill">{label_ai_confidence}: {confidence:.1%}</span>
                    </div>

                    <div class="analysis-section">
                        <h4>{label_detailed_analysis}</h4>
                        <div class="suitability-grid">
                            <div class="factor-card {get_ph_status_class(inputs['ph'])}">
                                <h5>{label_soil_ph_label}</h5>
                                <div class="value">{inputs['ph']:.1f}</div>
                                <div class="status">{get_ph_advice(inputs['ph'], lang)}</div>
                                <div class="recommendation">
                                    {get_ph_management_advice(inputs['ph'], lang).strip("<p>").strip("</p>")}
                                </div>
                            </div>

                            <div class="factor-card {get_temperature_status_class(inputs['temperature'])}">
                                <h5>{label_temp_label}</h5>
                                <div class="value">{inputs['temperature']}°C</div>
                                <div class="status">{get_temperature_advice(inputs['temperature'], lang)}</div>
                                <div class="recommendation">
                                    {get_temperature_management(inputs['temperature'], lang)}
                                </div>
                            </div>

                            <div class="factor-card {get_rainfall_status_class(inputs['rainfall'])}">
                                <h5>{label_rainfall_label}</h5>
                                <div class="value">{inputs['rainfall']} mm</div>
                                <div class="status">{get_rainfall_advice(inputs['rainfall'], lang)}</div>
                                <div class="recommendation">
                                    {get_rainfall_management(inputs['rainfall'], lang)}
                                </div>
                            </div>

                            <div class="factor-card {get_humidity_status_class(inputs['humidity'])}">
                                <h5>{label_humidity_label}</h5>
                                <div class="value">{inputs['humidity']}%</div>
                                <div class="status">{get_humidity_advice(inputs['humidity'], lang)}</div>
                                <div class="recommendation">
                                    {get_humidity_management(inputs['humidity'], lang)}
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="crop-requirements">
                        <h4>{label_optimal_conditions} {translated_prediction}</h4>
                        <div class="requirements-table">
                            <table class="table table-bordered">
                                <thead>
                                    <tr>
                                        <th>{label_factor}</th>
                                        <th>{label_optimal_range}</th>
                                        <th>{label_current_value}</th>
                                        <th>{label_status}</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>{label_soil_ph_label}</td>
                                        <td>6.0 - 7.5</td>
                                        <td>{inputs['ph']:.1f}</td>
                                        <td><span class="badge {ph_status.lower()}">{ph_status_text}</span></td>
                                    </tr>
                                    <tr>
                                        <td>{label_temp_label}</td>
                                        <td>20°C - 30°C</td>
                                        <td>{inputs['temperature']}°C</td>
                                        <td><span class="badge {get_temperature_status_class(inputs['temperature'])}">{get_temperature_status(inputs['temperature'], lang)}</span></td>
                                    </tr>
                                    <tr>
                                        <td>{label_rainfall_label}</td>
                                        <td>500 - 2000 mm</td>
                                        <td>{inputs['rainfall']} mm</td>
                                        <td><span class="badge {rainfall_status.lower()}">{rainfall_status_text}</span></td>
                                    </tr>
                                    <tr>
                                        <td>{label_humidity_label}</td>
                                        <td>30% - 90%</td>
                                        <td>{inputs['humidity']}%</td>
                                        <td><span class="badge {get_humidity_status_class(inputs['humidity'])}">{get_humidity_status(inputs['humidity'], lang)}</span></td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            """
        },
        'fertilizer_recommendation': {
            'title': f'🧪 {label_fert_rec}',
            'content': f"""
                <div class="fertilizer-analysis">
                    <div class="current-status">
                        <h3>{label_current_nutrient}</h3>
                        <div class="nutrient-grid">
                            <div class="nutrient-card {n_status.lower()}">
                                <div class="nutrient-header">
                                    <h4>{label_nitrogen}</h4>
                                    <span class="status-badge {n_status.lower()}">{n_status_text}</span>
                                </div>
                                <div class="nutrient-value">{inputs['N']} {t('ui.unit_kgha', lang)}</div>
                                <div class="optimal-range">{label_optimal}: 50-100 {t('ui.unit_kgha', lang)}</div>
                                <div class="recommendation">
                                    {get_detailed_nutrient_advice('N', inputs['N'], n_status, lang)}
                                </div>
                            </div>
                            <div class="nutrient-card {p_status.lower()}">
                                <div class="nutrient-header">
                                    <h4>{label_phosphorus}</h4>
                                    <span class="status-badge {p_status.lower()}">{p_status_text}</span>
                                </div>
                                <div class="nutrient-value">{inputs['P']} {t('ui.unit_kgha', lang)}</div>
                                <div class="optimal-range">{label_optimal}: 30-60 {t('ui.unit_kgha', lang)}</div>
                                <div class="recommendation">
                                    {get_detailed_nutrient_advice('P', inputs['P'], p_status, lang)}
                                </div>
                            </div>
                            <div class="nutrient-card {k_status.lower()}">
                                <div class="nutrient-header">
                                    <h4>{label_potassium}</h4>
                                    <span class="status-badge {k_status.lower()}">{k_status_text}</span>
                                </div>
                                <div class="nutrient-value">{inputs['K']} {t('ui.unit_kgha', lang)}</div>
                                <div class="optimal-range">{label_optimal}: 30-60 {t('ui.unit_kgha', lang)}</div>
                                <div class="recommendation">
                                    {get_detailed_nutrient_advice('K', inputs['K'], k_status, lang)}
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="fertilizer-recommendations">
                        <h3>{label_recommended_actions}</h3>
                        <div class="action-items">
                            <ul class="recommendation-list">
                                {generate_fertilizer_action_items(n_status, p_status, k_status, prediction, lang)}
                            </ul>
                        </div>
                    </div>

                    <div class="application-guidelines">
                        <h3>{label_application_guidelines}</h3>
                        <div class="guidelines-box">
                            <h4>{label_best_practices}</h4>
                            <ul>
                                <li>{t('timing.early_morning', lang)}</li>
                                <li>{t('timing.soil_moist', lang)}</li>
                                <li>{t('timing.incorporate', lang)}</li>
                                <li>{t('timing.split_doses', lang)}</li>
                            </ul>
                            
                            <h4>{label_timing}</h4>
                            <p>{label_for} {translated_prediction}:</p>
                            {get_crop_specific_timing(prediction, lang)}
                        </div>
                    </div>
                </div>
            """
        },
        'soil_health': {
            'title': f'🌿 {label_soil_health_mgmt}',
            'content': f"""
                <div class="soil-health-box">
                    <div class="ph-management">
                        <h3>{label_soil_ph_mgmt}</h3>
                        {get_ph_management_advice(inputs['ph'], lang)}
                        <div class="recommendations mt-4">
                            <h4>{label_ph_improvement}</h4>
                            <ul>
                                {f'<li>{t("advice.add_lime", lang)}</li>' if inputs['ph'] < 5.5 else ''}
                                {f'<li>{t("advice.add_sulfur", lang)}</li>' if inputs['ph'] > 7.5 else ''}
                                {f'<li>{t("advice.current_ph_optimal", lang)}</li>' if 5.5 <= inputs['ph'] <= 7.5 else ''}
                            </ul>
                        </div>
                    </div>
                    <div class="soil-management-practices mt-4">
                        <h3>{label_general_soil_mgmt}</h3>
                        <ul>
                            <li>{t('soil.crop_rotation', lang)}</li>
                            <li>{t('soil.add_organic', lang)}</li>
                            <li>{t('soil.drainage_systems', lang)}</li>
                            <li>{t('soil.monitor_moisture', lang)}</li>
                        </ul>
                    </div>
                </div>
            """
        }
    }
    
    if disease_info:
        label_disease_analysis = t('section.disease_analysis', lang)
        label_detection_results = t('disease.detection_results', lang)
        label_detected_disease = t('disease.detected_disease', lang)
        label_disease_confidence = t('disease.confidence', lang)
        label_severity = t('disease.severity', lang)
        label_treatment = t('disease.treatment', lang)
        
        advice_sections['disease_analysis'] = {
            'title': f'🔍 {label_disease_analysis}',
            'content': f"""
                <div class="disease-analysis-box">
                    <h3>{label_detection_results}</h3>
                    <div class="disease-info">
                        <p><strong>{label_detected_disease}:</strong> {disease_info['disease']}</p>
                        <p><strong>{label_disease_confidence}:</strong> {disease_info['confidence']:.1%}</p>
                        <p><strong>{label_severity}:</strong> <span class="badge bg-warning">{disease_info['severity']}</span></p>
                    </div>
                    <div class="treatment-info">
                        <h4>{label_treatment}</h4>
                        <p>{disease_info['treatment']}</p>
                    </div>
                </div>
            """
        }
    
    return advice_sections

def get_ph_advice(ph, lang=None):
    if lang is None:
        lang = get_user_language()
    
    if ph < 5.5:
        return t('advice.needs_acidic', lang)
    elif ph > 7.5:
        return t('advice.needs_alkaline', lang)
    else:
        return t('advice.optimal_range', lang)

def get_temperature_advice(temp, lang=None):
    if lang is None:
        lang = get_user_language()
    
    if temp < 15:
        return t('advice.too_cold', lang)
    elif temp > 35:
        return t('advice.too_hot', lang)
    else:
        return t('advice.optimal_range', lang)

def get_rainfall_advice(rainfall, lang=None):
    if lang is None:
        lang = get_user_language()
    
    if rainfall < 500:
        return t('advice.irrigation_needed', lang)
    elif rainfall > 2000:
        return t('advice.drainage_required', lang)
    else:
        return t('advice.adequate', lang)

def get_humidity_advice(humidity, lang=None):
    if lang is None:
        lang = get_user_language()
    
    if humidity < 30:
        return t('advice.too_dry', lang)
    elif humidity > 90:
        return t('advice.too_humid', lang)
    else:
        return t('advice.optimal_range', lang)

def get_nutrient_recommendation(nutrient, current, target, lang='en'):
    if target is None:
        return f"<p>{t('advice.no_target', lang)}</p>"
    
    diff = target - current
    if abs(diff) < 10:
        return f"<p>{t('advice.levels_appropriate', lang)}</p>"
    elif diff > 0:
        return f"<p>{t('advice.increase_by', lang)} {diff:.0f} {t('ui.unit_kgha', lang)}</p>"
    else:
        return f"<p>{t('advice.reduce_by', lang)} {abs(diff):.0f} {t('ui.unit_kgha', lang)}</p>"



def get_detailed_nutrient_advice(nutrient, value, status, lang='en'):
    if nutrient == 'N':
        if status == 'Low':
            advice = f"""
                <ul>
                    <li>{t('nutrient.apply_n_rich', lang)}</li>
                    <li>{t('nutrient.add_organic_matter', lang)}</li>
                    <li>{t('nutrient.nitrogen_fixing_crops', lang)}</li>
                </ul>
            """
        elif status == 'High':
            advice = f"""
                <ul>
                    <li>{t('nutrient.reduce_n_application', lang)}</li>
                    <li>{t('nutrient.heavy_n_feeding', lang)}</li>
                    <li>{t('nutrient.add_carbon_rich', lang)}</li>
                </ul>
            """
        else:
            advice = f"<p>{t('advice.maintain_practices', lang)}</p>"
    
    elif nutrient == 'P':
        if status == 'Low':
            advice = f"""
                <ul>
                    <li>{t('nutrient.apply_phosphate', lang)}</li>
                    <li>{t('nutrient.add_bone_meal', lang)}</li>
                    <li>{t('nutrient.maintain_ph_6_7', lang)}</li>
                </ul>
            """
        elif status == 'High':
            advice = f"""
                <ul>
                    <li>{t('nutrient.avoid_phosphorus', lang)}</li>
                    <li>{t('nutrient.monitor_water', lang)}</li>
                    <li>{t('nutrient.p_feeding_crops', lang)}</li>
                </ul>
            """
        else:
            advice = f"<p>{t('advice.maintain_practices', lang)}</p>"
    
    else:  # Potassium (K)
        if status == 'Low':
            advice = f"""
                <ul>
                    <li>{t('nutrient.apply_k_rich', lang)}</li>
                    <li>{t('nutrient.add_wood_ash', lang)}</li>
                    <li>{t('nutrient.maintain_drainage', lang)}</li>
                </ul>
            """
        elif status == 'High':
            advice = f"""
                <ul>
                    <li>{t('nutrient.avoid_potassium', lang)}</li>
                    <li>{t('nutrient.improve_drainage', lang)}</li>
                    <li>{t('nutrient.k_feeding_crops', lang)}</li>
                </ul>
            """
        else:
            advice = f"<p>{t('advice.maintain_practices', lang)}</p>"
    
    return advice

def generate_fertilizer_action_items(n_status, p_status, k_status, crop, lang='en'):
    actions = []
    
    # Load crop-specific requirements
    fertilizer_df = pd.read_csv('data/raw/fertilizer.csv')
    crop_req = fertilizer_df[fertilizer_df['Crop'].str.lower() == crop.lower()].iloc[0] if len(fertilizer_df[fertilizer_df['Crop'].str.lower() == crop.lower()]) > 0 else None
    
    # Add nutrient-specific recommendations
    if n_status == 'Low':
        actions.append(f"<li><strong>{t('ui.nitrogen', lang)}:</strong> {t('fert.n_split_doses', lang)}</li>")
    elif n_status == 'High':
        actions.append(f"<li><strong>{t('ui.nitrogen', lang)}:</strong> {t('fert.n_reduce', lang)}</li>")
    
    if p_status == 'Low':
        actions.append(f"<li><strong>{t('ui.phosphorus', lang)}:</strong> {t('fert.p_add', lang)}</li>")
    elif p_status == 'High':
        actions.append(f"<li><strong>{t('ui.phosphorus', lang)}:</strong> {t('fert.p_avoid', lang)}</li>")
    
    if k_status == 'Low':
        actions.append(f"<li><strong>{t('ui.potassium', lang)}:</strong> {t('fert.k_add', lang)}</li>")
    elif k_status == 'High':
        actions.append(f"<li><strong>{t('ui.potassium', lang)}:</strong> {t('fert.k_reduce', lang)}</li>")
    
    # Add crop-specific advice
    if crop_req is not None:
        translated_crop = translate(crop, lang, 'en') if lang != 'en' else crop
        actions.append(f"<li><strong>{t('fert.crop_specific_ratio', lang)} {translated_crop}:</strong> {crop_req['N']}-{crop_req['P']}-{crop_req['K']}</li>")
    
    # Add general recommendations
    actions.append(f"<li><strong>{t('fert.soil_testing', lang)}</strong></li>")
    actions.append(f"<li><strong>{t('fert.ph_management', lang)}</strong></li>")
    
    return "\n".join(actions)

def get_crop_specific_timing(crop, lang='en'):
    # Define timing recommendations using translation keys
    if crop.lower() == 'rice':
        timing = f"""
            <ul>
                <li>{t('timing.rice_base', lang)}</li>
                <li>{t('timing.rice_top1', lang)}</li>
                <li>{t('timing.rice_top2', lang)}</li>
            </ul>
        """
    elif crop.lower() == 'wheat':
        timing = f"""
            <ul>
                <li>{t('timing.wheat_base', lang)}</li>
                <li>{t('timing.wheat_top1', lang)}</li>
                <li>{t('timing.wheat_top2', lang)}</li>
            </ul>
        """
    elif crop.lower() == 'maize':
        timing = f"""
            <ul>
                <li>{t('timing.maize_base', lang)}</li>
                <li>{t('timing.maize_top1', lang)}</li>
                <li>{t('timing.maize_top2', lang)}</li>
            </ul>
        """
    else:
        timing = f"""
            <ul>
                <li>{t('timing.default_base', lang)}</li>
                <li>{t('timing.default_top', lang)}</li>
                <li>{t('timing.default_guidelines', lang)}</li>
            </ul>
        """
    
    return timing

def get_ph_status_class(ph):
    if ph < 5.5:
        return "warning"
    elif ph > 7.5:
        return "warning"
    return "success"

def get_temperature_status(temp, lang=None):
    if lang is None:
        lang = get_user_language()
    
    if temp < 15:
        return t('status.too_cold', lang)
    elif temp > 35:
        return t('status.too_hot', lang)
    else:
        return t('status.optimal', lang)

def get_temperature_status_class(temp):
    if temp < 15 or temp > 35:
        return "warning"
    return "success"

def get_temperature_management(temp, lang=None):
    if lang is None:
        lang = get_user_language()
    
    if temp < 15:
        return t('mgmt.too_cold', lang)
    elif temp > 35:
        return t('mgmt.too_hot', lang)
    else:
        return t('mgmt.maintain_practices', lang)

def get_rainfall_status_class(rainfall):
    if rainfall < 500:
        return "warning"
    elif rainfall > 2000:
        return "warning"
    return "success"

def get_rainfall_management(rainfall, lang=None):
    if lang is None:
        lang = get_user_language()
    
    if rainfall < 500:
        return t('mgmt.irrigation_conservation', lang)
    elif rainfall > 2000:
        return t('mgmt.drainage_raised_beds', lang)
    else:
        return t('mgmt.maintain_practices', lang)

def get_humidity_status(humidity, lang=None):
    if lang is None:
        lang = get_user_language()
    
    if humidity < 30:
        return t('status.too_dry', lang)
    elif humidity > 90:
        return t('status.too_humid', lang)
    else:
        return t('status.optimal', lang)

def get_humidity_status_class(humidity):
    if humidity < 30 or humidity > 90:
        return "warning"
    return "success"

def get_humidity_management(humidity, lang='en'):
    if humidity < 30:
        return t('mgmt.misting_systems', lang)
    elif humidity > 90:
        return t('mgmt.improve_ventilation', lang)
    else:
        return t('mgmt.maintain_practices', lang)

def get_ph_management_advice(ph, lang='en'):
    if ph < 5.5:
        text = t('advice.consider_lime', lang)
    elif ph > 7.5:
        text = t('advice.consider_sulfur', lang)
    else:
        text = t('advice.ph_optimal', lang)
    
    return f"<p>{text}</p>"


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


# ============================================================================
# CHATBOT ROUTES
# ============================================================================

@app.route('/chatbot')
def chatbot_page():
    """Render the chatbot interface."""
    # Set default language if not in session
    if 'language' not in session:
        session['language'] = 'en'
    return render_template('chatbot.html')


@app.route('/api/chatbot/init', methods=['POST'])
def chatbot_init():
    """Initialize a new chatbot session."""
    try:
        data = request.get_json() or {}
        language = session.get('language', 'en')
        provider = data.get('provider', 'auto')
        
        # Create chatbot instance and store in session
        # Note: We create a new instance for each init to avoid session serialization issues
        session['chatbot_config'] = {
            'language': language,
            'provider': provider,
            'initialized': True
        }
        
        # Get any context from previous analysis
        context = {}
        advice_sections = session.get('advice_sections')
        if advice_sections and 'crop_recommendation' in advice_sections:
            # Extract crop info if available
            context['has_analysis'] = True
        
        logger.info(f"Chatbot initialized for language: {language}, provider: {provider}")
        
        return jsonify({
            'status': 'success',
            'language': language,
            'provider': provider,
            'context': context,
            'welcome_message': t('chatbot.welcome', language)
        })
    
    except Exception as e:
        logger.error(f"Error initializing chatbot: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/chatbot/chat', methods=['POST'])
def chatbot_chat():
    """Handle chat messages."""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'status': 'error',
                'message': 'Message is required'
            }), 400
        
        # Get chatbot config from session
        chatbot_config = session.get('chatbot_config')
        if not chatbot_config or not chatbot_config.get('initialized'):
            return jsonify({
                'status': 'error',
                'message': 'Chatbot not initialized. Please refresh the page.'
            }), 400
        
        language = chatbot_config.get('language', 'en')
        provider = chatbot_config.get('provider', 'auto')
        
        # Get conversation history from session
        if 'chatbot_history' not in session:
            session['chatbot_history'] = []
        
        # Create chatbot instance
        chatbot = create_chatbot(language=language, provider=provider)
        
        # Set conversation history
        chatbot.conversation_history = session['chatbot_history']
        
        # Set context from user's analysis if available
        advice_sections = session.get('advice_sections')
        if advice_sections:
            context = {}
            # Extract crop recommendation
            crop_rec = advice_sections.get('crop_recommendation', {})
            if crop_rec:
                context['has_analysis'] = True
                # You can add more context extraction here
            chatbot.set_context(context)
        
        # Get response
        result = chatbot.chat(user_message)
        
        # Update session history
        session['chatbot_history'] = chatbot.conversation_history
        session.modified = True
        
        if result['success']:
            return jsonify({
                'status': 'success',
                'response': result['response'],
                'provider': result['provider'],
                'timestamp': result['timestamp']
            })
        else:
            return jsonify({
                'status': 'error',
                'message': result.get('error', 'Unknown error'),
                'response': result['response']
            }), 500
    
    except Exception as e:
        logger.error(f"Error in chatbot chat: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'response': t('chatbot.error', session.get('language', 'en'))
        }), 500


@app.route('/api/chatbot/clear', methods=['POST'])
def chatbot_clear():
    """Clear chat history."""
    try:
        session['chatbot_history'] = []
        session.modified = True
        
        return jsonify({
            'status': 'success',
            'message': 'Chat history cleared'
        })
    
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/chatbot/export', methods=['GET'])
def chatbot_export():
    """Export conversation history."""
    try:
        history = session.get('chatbot_history', [])
        config = session.get('chatbot_config', {})
        
        export_data = {
            'language': config.get('language', 'en'),
            'provider': config.get('provider', 'auto'),
            'history': history,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'data': export_data
        })
    
    except Exception as e:
        logger.error(f"Error exporting chat history: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    print("🚀 Starting AgroVision-AI application...")
    print("📦 Loading models and scaler...")
    if load_model_and_scaler():
        print("✅ Models loaded successfully!")
    else:
        print("⚠️ Models not found. The app will start, and the UI will show an error on analyze until models are provided in 'models/'.")
    print(f"🌐 Starting Flask server on http://0.0.0.0:5001 (Docker-friendly)")
    # Avoid multi-process reloader and extra threads on macOS to prevent native mutex issues
    debug_flag = os.getenv("FLASK_DEBUG", "0") == "1"
    # Bind to 0.0.0.0 so Docker port mapping exposes the service to host
    app.run(host="0.0.0.0", debug=debug_flag, port=5001, use_reloader=False, threaded=False)

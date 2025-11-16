# AgroVision-AI API Documentation

## Overview

AgroVision-AI provides a RESTful API for agricultural recommendations, plant disease detection, and fertilizer optimization.

**Base URL**: `http://localhost:5001`

**Version**: 1.0.0

## Authentication

Currently, the API does not require authentication. This may change in future versions.

## Endpoints

### 1. Home Page

**Endpoint**: `GET /`

**Description**: Renders the main input form

**Response**: HTML page

---

### 2. Analyze Agricultural Data

**Endpoint**: `POST /analyze`

**Description**: Submit agricultural data and optional plant image for comprehensive analysis

**Content-Type**: `multipart/form-data`

**Request Parameters**:

| Parameter | Type | Required | Range | Description |
|-----------|------|----------|-------|-------------|
| N | float | Yes | 0-140 | Nitrogen content (kg/ha) |
| P | float | Yes | 5-145 | Phosphorus content (kg/ha) |
| K | float | Yes | 5-205 | Potassium content (kg/ha) |
| temperature | float | Yes | 8-44 | Temperature (°C) |
| humidity | float | Yes | 14-100 | Humidity (%) |
| ph | float | Yes | 3.5-9.9 | Soil pH |
| rainfall | float | Yes | 20-300 | Rainfall (mm) |
| disease_image | file | No | - | Plant leaf image (JPG/PNG) |

**Example Request (curl)**:

```bash
curl -X POST http://localhost:5001/analyze \
  -F "N=90" \
  -F "P=42" \
  -F "K=43" \
  -F "temperature=20.87" \
  -F "humidity=82" \
  -F "ph=6.5" \
  -F "rainfall=202.9" \
  -F "disease_image=@/path/to/leaf.jpg"
```

**Response**: Redirects to `/results` with session data

**Error Response**:

```json
{
  "error": "Validation error message",
  "field": "parameter_name"
}
```

**Status Codes**:
- `302`: Redirect to results (success)
- `400`: Invalid input parameters
- `500`: Server error

---

### 3. View Results

**Endpoint**: `GET /results`

**Description**: Display analysis results from the previous analyze request

**Response**: HTML page with:
- Executive summary
- Crop recommendation
- Fertilizer recommendations
- Soil health analysis
- Disease detection results (if image was uploaded)
- AI-generated insights

**Example Response Structure**:

```html
<!-- Contains sections for: -->
<!-- - Executive Summary -->
<!-- - Crop Recommendation Analysis -->
<!-- - Fertilizer Recommendations -->
<!-- - Soil Health Management -->
<!-- - Disease Analysis (if applicable) -->
<!-- - AI Narrative -->
<!-- - LLM Alternative Recommendations -->
```

**Status Codes**:
- `200`: Success
- `302`: Redirect to home (no session data)

---

### 4. Validate Recommendations

**Endpoint**: `POST /validate_recs`

**Description**: Validate recommendations using LLM (requires prior analysis in session)

**Content-Type**: `application/json`

**Response**:

```json
{
  "validation": {
    "status": "success",
    "provider": "openai|anthropic|stub",
    "recommendations": [
      {
        "id": "crop_recommendation",
        "text": "Recommendation text",
        "valid": true,
        "reasoning": "Validation reasoning"
      }
    ]
  }
}
```

**Status Codes**:
- `200`: Success
- `400`: No analysis available in session
- `500`: Validation error

---

## Data Models

### Input Data Model

```python
{
  "N": float,           # Nitrogen (0-140 kg/ha)
  "P": float,           # Phosphorus (5-145 kg/ha)
  "K": float,           # Potassium (5-205 kg/ha)
  "temperature": float, # Temperature (8-44°C)
  "humidity": float,    # Humidity (14-100%)
  "ph": float,          # pH (3.5-9.9)
  "rainfall": float     # Rainfall (20-300 mm)
}
```

### Crop Recommendation Response

```python
{
  "prediction": str,      # Recommended crop name
  "confidence": float,    # Confidence score (0-1)
  "alternatives": [       # Alternative recommendations
    {
      "crop": str,
      "score": float
    }
  ]
}
```

### Disease Detection Response

```python
{
  "disease": str,        # Disease name
  "confidence": float,   # Confidence score (0-1)
  "severity": str,       # Low|Medium|High
  "treatment": str,      # Treatment recommendation
  "symptoms": [str]      # List of symptoms
}
```

### Fertilizer Recommendation Response

```python
{
  "N": {
    "current": float,
    "status": str,       # Low|Medium|High
    "recommendation": str
  },
  "P": {...},
  "K": {...},
  "application_timing": str,
  "guidelines": [str]
}
```

---

## Error Handling

### Error Response Format

```json
{
  "error": "Error message",
  "field": "field_name",
  "code": "ERROR_CODE"
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `INVALID_INPUT` | Input validation failed |
| `MISSING_FIELD` | Required field missing |
| `OUT_OF_RANGE` | Value outside valid range |
| `MODEL_ERROR` | Model prediction failed |
| `FILE_TOO_LARGE` | Uploaded file exceeds limit |
| `INVALID_FILE_TYPE` | Unsupported file format |

---

## Rate Limiting

Currently, there are no rate limits. This may be implemented in future versions.

---

## Examples

### Python Example

```python
import requests

# Prepare data
data = {
    'N': 90,
    'P': 42,
    'K': 43,
    'temperature': 20.87,
    'humidity': 82,
    'ph': 6.5,
    'rainfall': 202.9
}

# Optional: Add image
files = {}
if image_path:
    files['disease_image'] = open(image_path, 'rb')

# Make request
response = requests.post(
    'http://localhost:5001/analyze',
    data=data,
    files=files,
    allow_redirects=False
)

# Handle redirect
if response.status_code == 302:
    # Follow redirect to get results
    session = requests.Session()
    # Note: Session handling may be needed
    print("Analysis completed")
```

### JavaScript Example

```javascript
// Prepare form data
const formData = new FormData();
formData.append('N', 90);
formData.append('P', 42);
formData.append('K', 43);
formData.append('temperature', 20.87);
formData.append('humidity', 82);
formData.append('ph', 6.5);
formData.append('rainfall', 202.9);

// Optional: Add image
if (imageFile) {
    formData.append('disease_image', imageFile);
}

// Make request
fetch('http://localhost:5001/analyze', {
    method: 'POST',
    body: formData,
    credentials: 'include'  // Important for session
})
.then(response => {
    if (response.redirected) {
        window.location.href = response.url;
    }
})
.catch(error => console.error('Error:', error));
```

---

## Supported Crops

The system currently supports recommendations for the following crops:

- Rice
- Maize
- Chickpea
- Kidney Beans
- Pigeon Peas
- Moth Beans
- Mung Bean
- Black Gram
- Lentil
- Pomegranate
- Banana
- Mango
- Grapes
- Watermelon
- Muskmelon
- Apple
- Orange
- Papaya
- Coconut
- Cotton
- Jute
- Coffee

---

## Supported Plant Diseases

The disease detection model can identify:

### Apple Diseases
- Apple Scab
- Black Rot
- Cedar Apple Rust

### Tomato Diseases
- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Tomato Mosaic Virus
- Yellow Leaf Curl Virus

### Potato Diseases
- Early Blight
- Late Blight

### Pepper Diseases
- Bacterial Spot

And more (38+ disease classes total)

---

## Changelog

### v1.0.0 (Current)
- Initial API release
- Crop recommendation endpoint
- Disease detection endpoint
- Fertilizer recommendations
- LLM-enhanced insights

---

## Support

For API support, please:
- Open an issue on GitHub
- Check existing documentation
- Contact the maintainers

---

## Future Enhancements

Planned API features:
- [ ] RESTful JSON responses (in addition to HTML)
- [ ] Authentication and API keys
- [ ] Rate limiting
- [ ] Batch processing endpoint
- [ ] Webhook notifications
- [ ] Historical data tracking
- [ ] Weather API integration
- [ ] Mobile-optimized endpoints

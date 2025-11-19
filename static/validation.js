// Translation helper for validation
let validationTranslations = {};

async function loadValidationTranslations() {
    try {
        const resp = await fetch('/api/translate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                keys: [
                    'field.nitrogen', 'field.phosphorus', 'field.potassium',
                    'field.temperature', 'field.humidity', 'field.ph', 'field.rainfall',
                    'ui.soc', 'error.invalid_value', 'error.out_of_range', 'ui.error'
                ]
            })
        });
        const data = await resp.json();
        if (data.status === 'success') {
            validationTranslations = data.translations;
        }
    } catch (e) {
        console.error('Failed to load validation translations:', e);
    }
}

// Load translations when page loads
document.addEventListener('DOMContentLoaded', loadValidationTranslations);

// Form validation
document.getElementById('cropForm').addEventListener('submit', async function(event) {
    // Ensure translations are loaded
    if (Object.keys(validationTranslations).length === 0) {
        await loadValidationTranslations();
    }

    const inputFields = {
        'N': { min: 0, max: 200, key: 'field.nitrogen' },
        'P': { min: 5, max: 150, key: 'field.phosphorus' },
        'K': { min: 5, max: 200, key: 'field.potassium' },
        'temperature': { min: 15, max: 45, key: 'field.temperature' },
        'humidity': { min: 30, max: 100, key: 'field.humidity' },
        'ph': { min: 3, max: 10, key: 'field.ph' },
        'rainfall': { min: 100, max: 3000, key: 'field.rainfall' },
        'SOC': { min: 0.1, max: 3, key: 'ui.soc' }
    };

    let isValid = true;
    let errorMessage = '';

    for (const [fieldId, rules] of Object.entries(inputFields)) {
        const input = document.getElementById(fieldId);
        const value = parseFloat(input.value);
        const fieldName = validationTranslations[rules.key] || fieldId;

        if (isNaN(value)) {
            isValid = false;
            const invalidMsg = validationTranslations['error.invalid_value'] || 'Invalid value';
            errorMessage = `${invalidMsg}: ${fieldName}`;
            break;
        }

        if (value < rules.min || value > rules.max) {
            isValid = false;
            const rangeMsg = validationTranslations['error.out_of_range'] || 'must be between';
            errorMessage = `${fieldName} ${rangeMsg} ${rules.min} - ${rules.max}`;
            break;
        }
    }

    if (!isValid) {
        event.preventDefault();
        const errorLabel = validationTranslations['ui.error'] || 'Error';
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-danger alert-dismissible fade show mt-3';
        alertDiv.role = 'alert';
        alertDiv.innerHTML = `
            <strong>${errorLabel}:</strong> ${errorMessage}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Remove any existing alerts
        const existingAlerts = document.querySelectorAll('.alert');
        existingAlerts.forEach(alert => alert.remove());
        
        // Add new alert at the top of the form
        const form = document.getElementById('cropForm');
        form.insertBefore(alertDiv, form.firstChild);
    }
});
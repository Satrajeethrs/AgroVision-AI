// Form validation
document.getElementById('cropForm').addEventListener('submit', function(event) {
    const inputFields = {
        'N': { min: 0, max: 200, name: 'Nitrogen' },
        'P': { min: 5, max: 150, name: 'Phosphorus' },
        'K': { min: 5, max: 200, name: 'Potassium' },
        'temperature': { min: 15, max: 45, name: 'Temperature' },
        'humidity': { min: 30, max: 100, name: 'Humidity' },
        'ph': { min: 3, max: 10, name: 'pH' },
        'rainfall': { min: 100, max: 3000, name: 'Rainfall' },
        'SOC': { min: 0.1, max: 3, name: 'Soil Organic Carbon' }
    };

    let isValid = true;
    let errorMessage = '';

    for (const [fieldId, rules] of Object.entries(inputFields)) {
        const input = document.getElementById(fieldId);
        const value = parseFloat(input.value);

        if (isNaN(value)) {
            isValid = false;
            errorMessage = `Please enter a valid number for ${rules.name}`;
            break;
        }

        if (value < rules.min || value > rules.max) {
            isValid = false;
            errorMessage = `${rules.name} must be between ${rules.min} and ${rules.max}`;
            break;
        }
    }

    if (!isValid) {
        event.preventDefault();
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-danger alert-dismissible fade show mt-3';
        alertDiv.role = 'alert';
        alertDiv.innerHTML = `
            <strong>Error:</strong> ${errorMessage}
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
// Translation cache and helper function
let translationCache = {};

async function getTranslations(keys) {
    const uncachedKeys = keys.filter(k => !translationCache[k]);
    if (uncachedKeys.length > 0) {
        try {
            const resp = await fetch('/api/translate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({keys: uncachedKeys})
            });
            const data = await resp.json();
            if (data.status === 'success') {
                Object.assign(translationCache, data.translations);
            }
        } catch (e) {
            console.error('Translation fetch error:', e);
        }
    }
    return keys.map(k => translationCache[k] || k);
}

// Handle sidebar navigation
document.addEventListener('DOMContentLoaded', async function() {
    const sidebarButtons = document.querySelectorAll('.sidebar-btn');
    const contentSections = document.querySelectorAll('.content-section');

    // Function to switch active section
    function switchSection(sectionId) {
        // Update button states
        sidebarButtons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.section === sectionId);
        });

        // Update section visibility
        contentSections.forEach(section => {
            section.classList.toggle('active', section.id === sectionId);
        });
    }

    // Add click handlers to sidebar buttons
    sidebarButtons.forEach(button => {
        button.addEventListener('click', () => {
            const sectionId = button.dataset.section;
            switchSection(sectionId);
        });
    });

    // Preload translations for toggle button
    const [hideText, showText] = await getTranslations(['button.hide_raw', 'button.show_raw']);

    // Toggle raw AI response display
    const toggleBtn = document.getElementById('toggle-raw');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            const pre = document.getElementById('raw-output');
            if (!pre) return;
            if (pre.style.display === 'none') {
                pre.style.display = 'block';
                toggleBtn.innerText = hideText;
            } else {
                pre.style.display = 'none';
                toggleBtn.innerText = showText;
            }
        });
    }
});

// Validate recommendations on demand
document.addEventListener('DOMContentLoaded', async function() {
    const validateBtn = document.getElementById('btn-validate');
    const resultsDiv = document.getElementById('validation-results');
    if (!validateBtn || !resultsDiv) return;

    // Preload validation translations
    const [validatingText, validateText, idText, verdictText, confText, notesText] = 
        await getTranslations([
            'btn.validating',
            'button.validate_recommendations',
            'validation.id',
            'validation.verdict',
            'validation.confidence',
            'validation.notes'
        ]);

    validateBtn.addEventListener('click', async () => {
        validateBtn.disabled = true;
        validateBtn.innerText = validatingText;
        resultsDiv.innerHTML = '';
        try {
            const resp = await fetch('/validate_recs', {method: 'POST'});
            const data = await resp.json();
            if (data.error) {
                resultsDiv.innerHTML = `<div class="alert alert-warning">${data.error}</div>`;
            } else {
                // Render a simple table
                const rows = (data.results || []).map(r => `
                    <tr>
                        <td>${r.id || ''}</td>
                        <td>${r.verdict || ''}</td>
                        <td>${(r.confidence || 0).toFixed(2)}</td>
                        <td>${(r.notes || '').replace(/\n/g, '<br>')}</td>
                    </tr>`).join('');
                resultsDiv.innerHTML = `
                    <div class="table-responsive mt-2">
                        <table class="table table-sm table-bordered">
                            <thead><tr><th>${idText}</th><th>${verdictText}</th><th>${confText}</th><th>${notesText}</th></tr></thead>
                            <tbody>${rows}</tbody>
                        </table>
                    </div>`;
            }
        } catch (e) {
            resultsDiv.innerHTML = `<div class="alert alert-danger">${e}</div>`;
        } finally {
            validateBtn.disabled = false;
            validateBtn.innerText = validateText;
        }
    });
});

// Print current section
function printCurrentSection() {
    const currentSection = document.querySelector('.content-section.active');
    if (!currentSection) return;

    // Create a new window for printing
    const printWindow = window.open('', '_blank');
    const sectionTitle = currentSection.querySelector('.section-header h2').innerText;
    
    // Generate print content
    const printContent = `
        <!DOCTYPE html>
        <html>
        <head>
            <title>${sectionTitle}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { padding: 20px; }
                .section-content { margin-top: 20px; }
                @media print {
                    .no-print { display: none; }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h2 class="mb-4">${sectionTitle}</h2>
                <div class="section-content">
                    ${currentSection.querySelector('.section-content').innerHTML}
                </div>
            </div>
            <script>
                window.onload = function() { window.print(); window.close(); }
            </script>
        </body>
        </html>
    `;

    printWindow.document.write(printContent);
    printWindow.document.close();
}
// Handle sidebar navigation
document.addEventListener('DOMContentLoaded', function() {
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

    // Toggle raw AI response display
    const toggleBtn = document.getElementById('toggle-raw');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            const pre = document.getElementById('raw-output');
            if (!pre) return;
            if (pre.style.display === 'none') {
                pre.style.display = 'block';
                toggleBtn.innerText = 'Hide raw response';
            } else {
                pre.style.display = 'none';
                toggleBtn.innerText = 'Show raw response';
            }
        });
    }
});

// Validate recommendations on demand
document.addEventListener('DOMContentLoaded', function() {
    const validateBtn = document.getElementById('btn-validate');
    const resultsDiv = document.getElementById('validation-results');
    if (!validateBtn || !resultsDiv) return;

    validateBtn.addEventListener('click', async () => {
        validateBtn.disabled = true;
        validateBtn.innerText = 'Validating...';
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
                            <thead><tr><th>ID</th><th>Verdict</th><th>Conf</th><th>Notes</th></tr></thead>
                            <tbody>${rows}</tbody>
                        </table>
                    </div>`;
            }
        } catch (e) {
            resultsDiv.innerHTML = `<div class="alert alert-danger">${e}</div>`;
        } finally {
            validateBtn.disabled = false;
            validateBtn.innerText = 'Validate recommendations';
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
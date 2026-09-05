let chartInstance = null;
let explainChartInstance = null;

const form = document.getElementById('uploadForm');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('ecgFile');
const heaInput = document.getElementById('heaFile');
const heaPickBtn = document.getElementById('heaPickBtn');
const submitBtn = document.getElementById('submitBtn');
const fileStatus = document.getElementById('fileStatus');

const resultPanel = document.getElementById('resultPanel');
const pendingState = document.getElementById('pendingState');
const resultState = document.getElementById('resultState');
const errorState = document.getElementById('errorState');

function updateFileStatus() {
    const parts = [];
    if (fileInput.files[0]) parts.push(`<span class="filename">${fileInput.files[0].name}</span>`);
    if (heaInput.files[0]) parts.push(`<span class="filename">${heaInput.files[0].name}</span>`);
    fileStatus.innerHTML = parts.length ? parts.join(' + ') : '';
    submitBtn.disabled = !fileInput.files[0];
}

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});
['dragenter', 'dragover'].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
});
['dragleave', 'drop'].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); });
});
dropzone.addEventListener('drop', (e) => {
    const dropped = e.dataTransfer.files[0];
    if (dropped) {
        const dt = new DataTransfer();
        dt.items.add(dropped);
        fileInput.files = dt.files;
        updateFileStatus();
    }
});
fileInput.addEventListener('change', updateFileStatus);
heaInput.addEventListener('change', updateFileStatus);
heaPickBtn.addEventListener('click', () => heaInput.click());

function showPending() {
    resultPanel.hidden = false;
    pendingState.hidden = false;
    resultState.hidden = true;
    errorState.hidden = true;
    resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showError(message) {
    pendingState.hidden = true;
    resultState.hidden = true;
    errorState.hidden = false;
    errorState.textContent = message;
}

function showResult(data) {
    pendingState.hidden = true;
    errorState.hidden = true;
    resultState.hidden = false;

    const badge = document.getElementById('resultBadge');
    badge.textContent = data.prediction.toUpperCase();
    badge.className = `badge ${data.prediction}`;

    document.getElementById('processingTime').textContent = `${data.processing_time_ms} ms`;

    const pct = Math.round(data.confidence * 100);
    document.getElementById('confidenceBar').style.width = `${pct}%`;
    document.getElementById('confidenceLabel').textContent = `${pct}% model confidence`;

    document.getElementById('beatsAnalyzed').textContent = data.beats_analyzed;
    document.getElementById('modelArch').textContent = (data.model && data.model.architecture) || '-';

    const evalContext = document.getElementById('evalContext');
    const m = data.model || {};
    if (m.test_accuracy != null && m.test_n_records) {
        const pct = Math.round(m.test_accuracy * 100);
        evalContext.textContent = `Evaluated on ${m.test_n_records.toLocaleString()} held-out ECGs: ${pct}% record-level accuracy (research benchmark, not a guarantee for this ECG)`;
    } else {
        evalContext.textContent = '';
    }

    plotECG(data.signal);
    plotExplainability(data.explainability);
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!fileInput.files[0]) return;

    submitBtn.disabled = true;
    showPending();

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    if (heaInput.files[0]) formData.append('hea_file', heaInput.files[0]);

    try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const data = await response.json();
        if (response.ok) {
            showResult(data);
        } else {
            showError(data.error || 'Analysis failed.');
        }
    } catch (err) {
        showError('Network error: could not reach the analysis service.');
    } finally {
        submitBtn.disabled = !fileInput.files[0];
    }
});

function plotECG(signal) {
    const ctx = document.getElementById('ecgChart').getContext('2d');
    if (chartInstance) chartInstance.destroy();
    if (!signal || signal.length === 0) return;

    const gridColor = 'rgba(255,255,255,0.06)';
    const textColor = '#9aa4b2';

    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({ length: signal.length }, (_, i) => i),
            datasets: [{
                label: 'ECG Signal (Lead II)',
                data: signal,
                borderColor: '#3fd0c9',
                borderWidth: 1.5,
                fill: false,
                pointRadius: 0,
                tension: 0.15,
            }],
        },
        options: {
            responsive: true,
            animation: { duration: 400 },
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false, grid: { color: gridColor } },
                y: { ticks: { color: textColor }, grid: { color: gridColor } },
            },
        },
    });
}

function plotExplainability(explainability) {
    const wrap = document.getElementById('explainWrap');
    if (explainChartInstance) { explainChartInstance.destroy(); explainChartInstance = null; }
    if (!explainability || !explainability.saliency || explainability.saliency.length === 0) {
        wrap.hidden = true;
        return;
    }
    wrap.hidden = false;
    document.getElementById('explainNote').textContent = explainability.note;

    const ctx = document.getElementById('explainChart').getContext('2d');
    const beat = explainability.beat_window;
    const saliency = explainability.saliency;
    // Color each point by its attribution strength: dim teal (low) to
    // bright amber (high) - a heatmap-style overlay on the beat shape.
    const pointColors = saliency.map((s) => `rgba(255, ${Math.round(200 - s * 120)}, ${Math.round(80 + (1 - s) * 60)}, ${0.35 + s * 0.65})`);

    explainChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({ length: beat.length }, (_, i) => i),
            datasets: [{
                label: 'Beat window',
                data: beat,
                borderColor: 'rgba(154, 164, 178, 0.5)',
                borderWidth: 1,
                pointBackgroundColor: pointColors,
                pointBorderColor: pointColors,
                pointRadius: saliency.map((s) => 1 + s * 3),
                fill: false,
                tension: 0.15,
            }],
        },
        options: {
            responsive: true,
            animation: { duration: 300 },
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { display: false },
            },
        },
    });
}

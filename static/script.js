let chartInstance = null;

document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById('ecgFile');
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = 'Processing...';

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (response.ok) {
            resultDiv.innerHTML = `Prediction: <strong>${data.prediction}</strong>`;
            plotECG(data.signal);
        } else {
            resultDiv.innerHTML = `Error: ${data.error}`;
        }
    } catch (error) {
        resultDiv.innerHTML = `Error: ${error.message}`;
    }
});

function plotECG(signal) {
    const ctx = document.getElementById('ecgChart').getContext('2d');
    
    if (chartInstance) {
        chartInstance.destroy();
    }

    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: signal.length}, (_, i) => i),
            datasets: [{
                label: 'ECG Signal (Lead II)',
                data: signal,
                borderColor: 'blue',
                fill: false,
                pointRadius: 0
            }]
        },
        options: {
            scales: {
                x: { title: { display: true, text: 'Sample' } },
                y: { title: { display: true, text: 'Amplitude' } }
            }
        }
    });
}
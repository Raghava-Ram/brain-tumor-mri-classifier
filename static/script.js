document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    
    const uploadSection = document.querySelector('.upload-section');
    const previewSection = document.getElementById('previewSection');
    const imagePreview = document.getElementById('imagePreview');
    const loadingOverlay = document.getElementById('loadingOverlay');
    
    const resultsSection = document.getElementById('resultsSection');
    const mainPrediction = document.getElementById('mainPrediction');
    const confidenceFill = document.getElementById('confidenceFill');
    const mainConfidenceText = document.getElementById('mainConfidence');
    const probabilityList = document.getElementById('probabilityList');
    const resetBtn = document.getElementById('resetBtn');

    // --- Drag and Drop Events ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.remove('dragover'), false);
    });

    dropzone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        let dt = e.dataTransfer;
        let files = dt.files;
        if (files && files.length > 0) {
            handleFiles(files[0]);
        }
    }

    fileInput.addEventListener('change', function(e) {
        if (this.files && this.files.length > 0) {
            handleFiles(this.files[0]);
        }
    });

    function handleFiles(file) {
        if (!file.type.startsWith('image/')) {
            alert("Please upload an image file (JPG, PNG).");
            return;
        }

        // Setup UI for Loading
        uploadSection.classList.add('hidden');
        previewSection.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        loadingOverlay.classList.remove('hidden');

        // Render local preview
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
        }
        reader.readAsDataURL(file);

        // Upload to server
        uploadAndPredict(file);
    }

    async function uploadAndPredict(file) {
        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                // Try to extract a helpful error message from the API, if any
                let errorDetail = `Server returned ${response.status} ${response.statusText}`;
                try {
                    const errData = await response.json();
                    if (errData && (errData.detail || errData.message)) {
                        errorDetail = errData.detail || errData.message;
                    }
                } catch (_) {
                    // ignore JSON parse errors and fall back to default message
                }
                throw new Error(errorDetail);
            }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            console.error("Error doing prediction:", error);
            alert(`An error occurred during prediction.\n\nDetails: ${error.message}`);
            resetUI();
        }
    }

    function displayResults(data) {
        // Hide loading screen
        loadingOverlay.classList.add('hidden');
        resultsSection.classList.remove('hidden');

        // Populate main result
        mainPrediction.textContent = data.prediction.toUpperCase();
        
        // Color coding depending on result
        if (data.prediction.toLowerCase() === 'no tumor') {
            mainPrediction.style.color = 'var(--accent-green)';
            mainPrediction.style.textShadow = '0 0 20px rgba(63, 185, 80, 0.3)';
        } else {
            mainPrediction.style.color = 'var(--danger)';
            mainPrediction.style.textShadow = '0 0 20px rgba(248, 81, 73, 0.3)';
        }

        // Animate confidence bar
        setTimeout(() => {
            confidenceFill.style.width = data.confidence.toFixed(1) + '%';
        }, 100);
        mainConfidenceText.textContent = data.confidence.toFixed(2) + '%';

        // Populate detailed list
        probabilityList.innerHTML = '';
        for (const [className, prob] of Object.entries(data.probabilities)) {
            const li = document.createElement('li');
            li.className = 'prob-item';
            
            const labelSpan = document.createElement('span');
            labelSpan.className = 'prob-label';
            labelSpan.textContent = className;
            
            const valueSpan = document.createElement('span');
            valueSpan.className = 'prob-value';
            valueSpan.textContent = prob.toFixed(2) + '%';
            
            li.appendChild(labelSpan);
            li.appendChild(valueSpan);
            probabilityList.appendChild(li);
        }
    }

    function resetUI() {
        fileInput.value = '';
        uploadSection.classList.remove('hidden');
        previewSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        imagePreview.src = '';
        confidenceFill.style.width = '0%';
    }

    resetBtn.addEventListener('click', resetUI);
});

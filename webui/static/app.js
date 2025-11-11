// State
let selectedModel = null;
let uploadedVideo = null;
let ws = null;
let isInferenceRunning = false;

// DOM Elements
const modelSelect = document.getElementById('modelSelect');
const refreshModelsBtn = document.getElementById('refreshModels');
const videoUpload = document.getElementById('videoUpload');
const fileUploadDisplay = document.getElementById('fileUploadDisplay');
const startInferenceBtn = document.getElementById('startInference');
const stopInferenceBtn = document.getElementById('stopInference');
const inferenceFrame = document.getElementById('inferenceFrame');
const videoContainer = document.getElementById('videoContainer');
const confThreshold = document.getElementById('confThreshold');
const nmsThreshold = document.getElementById('nmsThreshold');
const confValue = document.getElementById('confValue');
const nmsValue = document.getElementById('nmsValue');

// Stats elements
const currentFpsEl = document.getElementById('currentFps');
const inferenceTimeEl = document.getElementById('inferenceTime');
const avgInferenceEl = document.getElementById('avgInference');
const progressEl = document.getElementById('progress');
const progressFill = document.getElementById('progressFill');
const frameCounter = document.getElementById('frameCounter');

// Info elements
const modelInfo = document.getElementById('modelInfo');
const videoInfo = document.getElementById('videoInfo');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadModels();
    setupEventListeners();
});

// Load available models
async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        modelSelect.innerHTML = '<option value="">Select a model...</option>';
        
        data.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.path;
            option.textContent = model.name;
            option.dataset.size = model.size;
            modelSelect.appendChild(option);
        });
        
        showStatus('Models loaded successfully', 'success');
    } catch (error) {
        showStatus('Failed to load models: ' + error.message, 'error');
    }
}

// Event Listeners
function setupEventListeners() {
    refreshModelsBtn.addEventListener('click', loadModels);
    
    modelSelect.addEventListener('change', (e) => {
        selectedModel = e.target.value;
        if (selectedModel) {
            const selectedOption = e.target.options[e.target.selectedIndex];
            modelInfo.innerHTML = `
                <strong>Selected:</strong> ${selectedOption.textContent}<br>
                <strong>Size:</strong> ${selectedOption.dataset.size}
            `;
            modelInfo.classList.add('active');
        } else {
            modelInfo.classList.remove('active');
        }
        updateStartButton();
    });
    
    videoUpload.addEventListener('change', handleVideoUpload);
    
    confThreshold.addEventListener('input', (e) => {
        confValue.textContent = e.target.value;
    });
    
    nmsThreshold.addEventListener('input', (e) => {
        nmsValue.textContent = e.target.value;
    });
    
    startInferenceBtn.addEventListener('click', startInference);
    stopInferenceBtn.addEventListener('click', stopInference);
    
    // Drag and drop
    const fileUploadWrapper = document.querySelector('.file-upload-wrapper');
    
    fileUploadWrapper.addEventListener('dragover', (e) => {
        e.preventDefault();
        fileUploadDisplay.style.borderColor = 'var(--primary)';
    });
    
    fileUploadWrapper.addEventListener('dragleave', (e) => {
        e.preventDefault();
        fileUploadDisplay.style.borderColor = 'var(--border)';
    });
    
    fileUploadWrapper.addEventListener('drop', (e) => {
        e.preventDefault();
        fileUploadDisplay.style.borderColor = 'var(--border)';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            videoUpload.files = files;
            handleVideoUpload({ target: { files } });
        }
    });
}

// Handle video upload
async function handleVideoUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        showStatus('Uploading video...', 'info');
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Upload failed');
        }
        
        const data = await response.json();
        uploadedVideo = data;
        
        // Update UI
        fileUploadDisplay.innerHTML = `
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
            </svg>
            <p style="color: var(--success);">Video uploaded successfully</p>
            <span>${file.name}</span>
        `;
        
        videoInfo.innerHTML = `
            <strong>File:</strong> ${data.filename}<br>
            <strong>Resolution:</strong> ${data.width}x${data.height}<br>
            <strong>FPS:</strong> ${data.fps.toFixed(2)}<br>
            <strong>Frames:</strong> ${data.frames}<br>
            <strong>Duration:</strong> ${data.duration}
        `;
        videoInfo.classList.add('active');
        
        showStatus('Video uploaded successfully', 'success');
        updateStartButton();
        
    } catch (error) {
        showStatus('Upload failed: ' + error.message, 'error');
    }
}

// Update start button state
function updateStartButton() {
    startInferenceBtn.disabled = !(selectedModel && uploadedVideo);
}

// Start inference
async function startInference() {
    if (!selectedModel || !uploadedVideo) {
        showStatus('Please select a model and upload a video', 'error');
        return;
    }
    
    isInferenceRunning = true;
    startInferenceBtn.disabled = true;
    stopInferenceBtn.disabled = false;
    
    // Show inference frame
    videoContainer.querySelector('.placeholder')?.remove();
    inferenceFrame.style.display = 'block';
    
    // Create WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/inference`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        showStatus('Starting inference...', 'info');
        
        // Send configuration
        ws.send(JSON.stringify({
            video_path: uploadedVideo.path,
            model_path: selectedModel,
            conf_threshold: parseFloat(confThreshold.value),
            nms_threshold: parseFloat(nmsThreshold.value)
        }));
    };
    
    let lastFrameTime = Date.now();
    let frameCount = 0;
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.error) {
            showStatus('Error: ' + data.error, 'error');
            stopInference();
            return;
        }
        
        if (data.complete) {
            showStatus(`Inference complete! Processed ${data.total_frames} frames`, 'success');
            stopInference();
            return;
        }
        
        // Update frame
        if (data.frame) {
            inferenceFrame.src = `data:image/jpeg;base64,${data.frame}`;
            frameCounter.textContent = `Frame ${data.frame_idx + 1}/${data.total_frames}`;
            
            // Calculate FPS
            const now = Date.now();
            const timeDiff = now - lastFrameTime;
            lastFrameTime = now;
            const fps = 1000 / timeDiff;
            
            // Update stats
            currentFpsEl.textContent = fps.toFixed(1);
            inferenceTimeEl.textContent = `${data.inference_time} ms`;
            avgInferenceEl.textContent = `${data.avg_inference_time} ms`;
            progressEl.textContent = `${data.progress.toFixed(1)}%`;
            progressFill.style.width = `${data.progress}%`;
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        showStatus('Connection error', 'error');
        stopInference();
    };
    
    ws.onclose = () => {
        if (isInferenceRunning) {
            showStatus('Connection closed', 'info');
            stopInference();
        }
    };
}

// Stop inference
function stopInference() {
    isInferenceRunning = false;
    startInferenceBtn.disabled = false;
    stopInferenceBtn.disabled = true;
    
    if (ws) {
        ws.close();
        ws = null;
    }
}

// Show status message
function showStatus(message, type = 'info') {
    const statusMessages = document.getElementById('statusMessages');
    
    const messageEl = document.createElement('div');
    messageEl.className = `status-message ${type}`;
    
    const icon = type === 'success' 
        ? '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>'
        : type === 'error'
        ? '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>'
        : '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>';
    
    messageEl.innerHTML = `
        ${icon}
        <span>${message}</span>
    `;
    
    statusMessages.appendChild(messageEl);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        messageEl.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => messageEl.remove(), 300);
    }, 5000);
}

// Reset stats
function resetStats() {
    currentFpsEl.textContent = '0';
    inferenceTimeEl.textContent = '0 ms';
    avgInferenceEl.textContent = '0 ms';
    progressEl.textContent = '0%';
    progressFill.style.width = '0%';
    frameCounter.textContent = 'Frame 0/0';
}

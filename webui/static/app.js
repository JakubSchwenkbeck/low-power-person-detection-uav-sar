let selectedModel = null, uploadedVideo = null, ws = null, isRunning = false;

const $ = id => document.getElementById(id);

async function loadModels() {
    const data = await fetch('/api/models').then(r => r.json());
    $('modelSelect').innerHTML = '<option value="">Select a model...</option>' + 
        data.models.map(m => `<option value="${m.path}" data-size="${m.size}">${m.name}</option>`).join('');
    showStatus('Models loaded', 'success');
}

async function handleVideoUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    showStatus('Uploading...', 'info');
    const data = await fetch('/api/upload', { method: 'POST', body: formData }).then(r => r.json());
    uploadedVideo = data;
    
    $('fileUploadDisplay').innerHTML = `
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
        </svg>
        <p style="color: var(--success);">Video uploaded</p>
        <span>${file.name}</span>
    `;
    
    $('videoInfo').innerHTML = `
        <strong>File:</strong> ${data.filename}<br>
        <strong>Resolution:</strong> ${data.width}x${data.height}<br>
        <strong>FPS:</strong> ${data.fps.toFixed(2)}<br>
        <strong>Frames:</strong> ${data.frames}<br>
        <strong>Duration:</strong> ${data.duration}
    `;
    $('videoInfo').classList.add('active');
    showStatus('Video uploaded', 'success');
    updateStartButton();
}

function updateStartButton() {
    $('startInference').disabled = !(selectedModel && uploadedVideo);
}

function startInference() {
    if (!selectedModel || !uploadedVideo) return;
    
    isRunning = true;
    $('startInference').disabled = true;
    $('stopInference').disabled = false;
    $('videoContainer').querySelector('.placeholder')?.remove();
    $('inferenceFrame').style.display = 'block';
    
    const benchmarkMode = $('benchmarkMode').checked;
    
    const benchmarkCards = ['memoryCard', 'cpuCard', 'tempCard', 'energyCard'];
    benchmarkCards.forEach(id => {
        $(id).style.display = benchmarkMode ? 'flex' : 'none';
    });
    
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws/inference`);
    
    ws.onopen = () => {
        const config = {
            video_path: uploadedVideo.path,
            model_path: selectedModel,
            conf_threshold: parseFloat($('confThreshold').value),
            nms_threshold: parseFloat($('nmsThreshold').value),
            benchmark: benchmarkMode
        };
        
        showStatus(benchmarkMode ? 'Starting inference with benchmarking...' : 'Starting inference...', 'info');
        ws.send(JSON.stringify(config));
    };
    
    let lastTime = Date.now();
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.error) {
            showStatus('Error: ' + data.error, 'error');
            stopInference();
            return;
        }
        
        if (data.complete) {
            let msg = data.frames_with_detections 
                ? `Processed ${data.total_frames} frames - ${data.frames_with_detections} had detections`
                : `Processed ${data.total_frames} frames - No detections`;
            
            if (data.benchmark) {
                msg += `\n\nBenchmark Results:\n` +
                       `Avg Memory: ${data.benchmark.avg_memory_usage_MiB} MiB\n` +
                       `Avg CPU: ${data.benchmark.avg_cpu_usage_percent}%\n` +
                       `Avg Energy: ${data.benchmark.avg_energy_consumption_W} W`;
                if (data.benchmark.avg_temperature_C) {
                    msg += `\nAvg Temp: ${data.benchmark.avg_temperature_C}°C`;
                }
                
                if (data.benchmark_file) {
                    generatePlot(data.benchmark_file);
                }
            }
            
            showStatus(msg, data.frames_with_detections > 0 ? 'success' : 'info');
            stopInference();
            return;
        }
        
        if (data.frame) {
            $('inferenceFrame').src = `data:image/jpeg;base64,${data.frame}`;
            $('frameCounter').textContent = `Frame ${data.frame_idx + 1}/${data.total_frames}`;
            
            const now = Date.now();
            const fps = 1000 / (now - lastTime);
            lastTime = now;
            
            $('currentFps').textContent = fps.toFixed(1);
            $('inferenceTime').textContent = `${data.inference_time} ms`;
            $('avgInference').textContent = `${data.avg_inference_time} ms`;
            $('progress').textContent = `${data.progress.toFixed(1)}%`;
            $('progressFill').style.width = `${data.progress}%`;
            
            // Update benchmark metrics if available
            if (data.memory_usage) {
                $('memoryUsage').textContent = `${data.memory_usage} MiB`;
            }
            if (data.cpu_usage) {
                $('cpuUsage').textContent = `${data.cpu_usage}%`;
            }
            if (data.temperature) {
                $('temperature').textContent = `${data.temperature}°C`;
            }
            if (data.energy) {
                $('energy').textContent = `${data.energy} W`;
            }
        }
    };
    
    ws.onerror = () => stopInference();
    ws.onclose = () => isRunning && stopInference();
}

async function generatePlot(benchmarkFile) {
    try {
        showStatus('Generating benchmark plots...', 'info');
        const response = await fetch('/api/plot_benchmark', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ benchmark_file: benchmarkFile })
        });
        
        if (!response.ok) {
            const error = await response.text();
            showStatus('Failed to generate plots: ' + error, 'error');
            return;
        }
        
        const data = await response.json();
        
        if (data.success) {
            showStatus('Benchmark plots generated!', 'success');
            displayBenchmarkPlot(data.plot_file, benchmarkFile);
        }
    } catch (error) {
        showStatus('Failed to generate plots: ' + error.message, 'error');
    }
}

function displayBenchmarkPlot(plotFile, jsonFile) {
    const resultsDiv = $('benchmarkResults');
    const contentDiv = $('benchmarkContent');
    
    const plotPath = '/plots/' + plotFile.split('/').pop();
    const jsonPath = '/benchmarks/' + jsonFile.split('/').pop();
    
    contentDiv.innerHTML = `
        <div style="margin: 1rem 0;">
            <p style="margin-bottom: 1rem;">
                <strong>Benchmark Data:</strong> ${jsonFile.split('/').pop()}<br>
                <strong>Plot Generated:</strong> ${plotFile.split('/').pop()}
            </p>
            <img src="${plotPath}" alt="Benchmark Plot" style="width: 100%; border-radius: 12px; border: 1px solid var(--border);" 
                 onerror="console.error('Failed to load image:', this.src)">
            <div style="margin-top: 1rem; display: flex; gap: 1rem;">
                <a href="${plotPath}" download class="btn btn-primary" style="flex: 1; text-decoration: none;">
                    📥 Download Plot
                </a>
                <a href="${jsonPath}" download class="btn btn-secondary" style="flex: 1; text-decoration: none;">
                    📄 Download JSON
                </a>
            </div>
        </div>
    `;
    
    resultsDiv.style.display = 'block';
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

function stopInference() {
    isRunning = false;
    $('startInference').disabled = false;
    $('stopInference').disabled = true;
    if (ws) {
        ws.close();
        ws = null;
    }
}

function showStatus(message, type = 'info') {
    const el = document.createElement('div');
    el.className = `status-message ${type}`;
    const icons = {
        success: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>',
        error: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>',
        info: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>'
    };
    el.innerHTML = `${icons[type]}<span>${message}</span>`;
    $('statusMessages').appendChild(el);
    setTimeout(() => {
        el.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => el.remove(), 300);
    }, 5000);
}

document.addEventListener('DOMContentLoaded', () => {
    loadModels();
    
    $('refreshModels').addEventListener('click', loadModels);
    $('modelSelect').addEventListener('change', (e) => {
        selectedModel = e.target.value;
        if (selectedModel) {
            const opt = e.target.options[e.target.selectedIndex];
            $('modelInfo').innerHTML = `<strong>Selected:</strong> ${opt.textContent}<br><strong>Size:</strong> ${opt.dataset.size}`;
            $('modelInfo').classList.add('active');
        } else {
            $('modelInfo').classList.remove('active');
        }
        updateStartButton();
    });
    $('videoUpload').addEventListener('change', handleVideoUpload);
    $('confThreshold').addEventListener('input', (e) => $('confValue').textContent = e.target.value);
    $('nmsThreshold').addEventListener('input', (e) => $('nmsValue').textContent = e.target.value);
    $('startInference').addEventListener('click', startInference);
    $('stopInference').addEventListener('click', stopInference);
    
    const wrapper = document.querySelector('.file-upload-wrapper');
    wrapper.addEventListener('dragover', (e) => {
        e.preventDefault();
        $('fileUploadDisplay').style.borderColor = 'var(--primary)';
    });
    wrapper.addEventListener('dragleave', (e) => {
        e.preventDefault();
        $('fileUploadDisplay').style.borderColor = 'var(--border)';
    });
    wrapper.addEventListener('drop', (e) => {
        e.preventDefault();
        $('fileUploadDisplay').style.borderColor = 'var(--border)';
        if (e.dataTransfer.files.length > 0) {
            $('videoUpload').files = e.dataTransfer.files;
            handleVideoUpload({ target: { files: e.dataTransfer.files } });
        }
    });
});

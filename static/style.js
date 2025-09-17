document.addEventListener('DOMContentLoaded', function() {
    // Control Buttons
    const startRTSPButton = document.getElementById('startRTSPButton');
    const processVideoButton = document.getElementById('processVideoButton');
    const stopButton = document.getElementById('stopButton');
    
    // Inputs
    const rtspUrlInput = document.getElementById('rtspUrl');
    const cameraIdInput = document.getElementById('cameraId');
    const videoFileInput = document.getElementById('videoFile');
    
    // Status Display
    const statusMessage = document.getElementById('statusMessage');

    let detectionInterval;

    // --- Event Listeners ---
    startRTSPButton.addEventListener('click', startRTSP);
    processVideoButton.addEventListener('click', processVideo);
    stopButton.addEventListener('click', stopProcess);

    // --- Core Functions ---
    function startRTSP() {
        const rtspUrl = rtspUrlInput.value.trim();
        const cameraId = cameraIdInput.value.trim();
        if (!rtspUrl || !cameraId) {
            updateStatus('Please provide both an RTSP URL and a Camera ID.', 'danger');
            return;
        }

        fetch('/start_rtsp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rtsp_url: rtspUrl, camera_id: cameraId }),
        })
        .then(handleResponse)
        .catch(handleError);
    }

    function processVideo() {
        const videoFile = videoFileInput.files[0];
        if (!videoFile) {
            updateStatus('Please select a video file to process.', 'danger');
            return;
        }

        const formData = new FormData();
        formData.append('video', videoFile);

        updateStatus('Uploading and starting video processing...', 'info');
        toggleAllControls(true); // Disable all inputs

        fetch('/process_video', {
            method: 'POST',
            body: formData,
        })
        .then(handleResponse)
        .catch(handleError);
    }

    function stopProcess() {
        fetch('/stop', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    updateStatus(data.message, 'info');
                    toggleAllControls(false); // Re-enable all inputs
                    clearInterval(detectionInterval);
                } else {
                    updateStatus(data.message, 'warning');
                }
            })
            .catch(handleError);
    }

    function fetchDetections() {
        fetch('/latest_detections')
            .then(response => response.json())
            .then(data => {
                updateDetectionCard('garbage', data.garbage_overflow);
                updateDetectionCard('animals', data.stray_animals);
                updateDetectionCard('roads', data.damaged_roads);
            })
            .catch(error => console.error('Error fetching detections:', error));
    }
    
    // --- Helper & UI Functions ---
    function handleResponse(response) {
        return response.json().then(data => {
            if (data.status === 'success') {
                updateStatus(data.message, 'success');
                toggleAllControls(true); // A process has started
                // Start polling for detections every 5 seconds
                detectionInterval = setInterval(fetchDetections, 5000);
                fetchDetections(); // Initial fetch
            } else {
                updateStatus(data.message, 'danger');
                toggleAllControls(false); // Process failed to start
            }
        });
    }

    function handleError(error) {
        console.error('Error:', error);
        updateStatus('An unexpected error occurred. Check the console.', 'danger');
        toggleAllControls(false);
    }

    function updateStatus(message, type) {
        // Simple type mapping to bootstrap classes
        const alertClass = `alert-${type}`
        statusMessage.textContent = `Status: ${message}`;
        // Add a bit of styling, remove old styles
        statusMessage.classList.remove('alert-success', 'alert-danger', 'alert-info', 'alert-warning');
        // statusMessage.classList.add(alertClass); // Optional: if you want colored backgrounds
    }

    function toggleAllControls(isProcessing) {
        startRTSPButton.disabled = isProcessing;
        processVideoButton.disabled = isProcessing;
        rtspUrlInput.disabled = isProcessing;
        cameraIdInput.disabled = isProcessing;
        videoFileInput.disabled = isProcessing;
        stopButton.disabled = !isProcessing;
    }

    function updateDetectionCard(category, data) {
        const card = document.getElementById(`${category}_card`);
        const img = document.getElementById(`${category}_img`);
        const cam = document.getElementById(`${category}_cam`);
        const time = document.getElementById(`${category}_time`);
        const date = document.getElementById(`${category}_date`);
        const loc = document.getElementById(`${category}_loc`);

        if (data) {
            img.src = `${data.image_url}?t=${new Date().getTime()}`;
            cam.textContent = data.camera_id;
            time.textContent = data.time;
            date.textContent = data.date;
            loc.textContent = `Lat: ${data.latitude}, Lon: ${data.longitude}`;
            card.classList.add('detected');
        }
    }
});
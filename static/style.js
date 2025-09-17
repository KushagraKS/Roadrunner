document.addEventListener('DOMContentLoaded', function() {
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const rtspUrlInput = document.getElementById('rtspUrl');
    const cameraIdInput = document.getElementById('cameraId');
    const statusMessage = document.getElementById('statusMessage');

    let detectionInterval;

    // --- Event Listeners ---

    startButton.addEventListener('click', () => {
        const rtspUrl = rtspUrlInput.value.trim();
        const cameraId = cameraIdInput.value.trim();

        if (!rtspUrl || !cameraId) {
            updateStatus('Please provide both an RTSP URL and a Camera ID.', 'danger');
            return;
        }

        startMonitoring(rtspUrl, cameraId);
    });

    stopButton.addEventListener('click', () => {
        stopMonitoring();
    });

    // --- API Call Functions ---

    function startMonitoring(rtspUrl, cameraId) {
        fetch('/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rtsp_url: rtspUrl, camera_id: cameraId }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                updateStatus(data.message, 'success');
                toggleControls(true);
                // Start polling for detections every 5 seconds
                detectionInterval = setInterval(fetchDetections, 5000);
                fetchDetections(); // Initial fetch
            } else {
                updateStatus(data.message, 'danger');
            }
        })
        .catch(error => {
            console.error('Error starting stream:', error);
            updateStatus('Failed to start monitoring. Check console for details.', 'danger');
        });
    }

    function stopMonitoring() {
        fetch('/stop', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                updateStatus(data.message, 'info');
                toggleControls(false);
                clearInterval(detectionInterval); // Stop polling
            } else {
                updateStatus(data.message, 'warning');
            }
        })
        .catch(error => {
            console.error('Error stopping stream:', error);
            updateStatus('Failed to stop monitoring.', 'danger');
        });
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

    // --- UI Update Functions ---

    function updateDetectionCard(category, data) {
        const card = document.getElementById(`${category}_card`);
        const img = document.getElementById(`${category}_img`);
        const cam = document.getElementById(`${category}_cam`);
        const time = document.getElementById(`${category}_time`);
        const date = document.getElementById(`${category}_date`);

        if (data) {
            // Add a cache-busting query parameter to force image reload
            img.src = `${data.image_url}?t=${new Date().getTime()}`;
            cam.textContent = data.camera_id;
            time.textContent = data.time;
            date.textContent = data.date;
            card.classList.add('detected');
        }
        // You can add an 'else' block here to reset the card if you want detections to disappear after some time
    }

    function updateStatus(message, type) {
        statusMessage.textContent = message;
        statusMessage.className = `mt-3 alert alert-${type}`;
    }

    function toggleControls(isMonitoring) {
        startButton.disabled = isMonitoring;
        stopButton.disabled = !isMonitoring;
        rtspUrlInput.disabled = isMonitoring;
        cameraIdInput.disabled = isMonitoring;
    }
});
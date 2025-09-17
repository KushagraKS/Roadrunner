import os
import cv2
import time
import threading
import random
from datetime import datetime
from flask import Flask, render_template, jsonify, request

# --- Configuration ---
# Create a Flask web application instance
app = Flask(__name__)
# Suppress console output for GET/POST requests for a cleaner log
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- Global State ---
# This dictionary will hold the background thread for processing the stream
processing_thread = None
# A flag to signal the thread to stop
stop_processing_flag = threading.Event()

# --- Placeholder Detection Models ---
# IMPORTANT: Replace these with your actual model inference logic.
# They currently return a random True/False to simulate detection.

def detect_garbage_overflow(frame):
    """Simulates garbage overflow detection."""
    return random.choice([True, False])

def detect_stray_animals(frame):
    """Simulates stray animal detection."""
    return random.choice([True, False])

def detect_damaged_roads(frame):
    """Simulates damaged road detection."""
    return random.choice([True, False])

# --- Core Video Processing Logic ---

def process_rtsp_stream(rtsp_url, camera_id):
    """
    Connects to an RTSP stream, captures frames every 5 seconds,
    and processes them in 30-second batches.
    """
    print(f"🚀 Starting stream processing for Camera ID: {camera_id}...")
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print(f"❌ Error: Could not open RTSP stream at {rtsp_url}")
        return

    frames_batch = []
    last_capture_time = time.time()
    last_processing_time = time.time()

    while not stop_processing_flag.is_set():
        current_time = time.time()

        # Capture a frame every 5 seconds
        if current_time - last_capture_time >= 5:
            ret, frame = cap.read()
            if not ret:
                print("❌ Warning: Failed to grab frame. Stream may have ended.")
                time.sleep(2) # Wait before retrying
                # Attempt to reconnect
                cap.release()
                cap = cv2.VideoCapture(rtsp_url)
                continue

            frames_batch.append(frame)
            last_capture_time = current_time
            # print(f"📸 Snapshot captured. Total in batch: {len(frames_batch)}")

        # Process the batch every 30 seconds
        if current_time - last_processing_time >= 30 and frames_batch:
            print(f"\n⏳ Processing a batch of {len(frames_batch)} frames...")
            
            problematic_frames = {
                'garbage_overflow': None,
                'stray_animals': None,
                'damaged_roads': None
            }

            # Run detection on each frame in the batch
            for frame_to_process in frames_batch:
                if detect_garbage_overflow(frame_to_process):
                    problematic_frames['garbage_overflow'] = frame_to_process
                if detect_stray_animals(frame_to_process):
                    problematic_frames['stray_animals'] = frame_to_process
                if detect_damaged_roads(frame_to_process):
                    problematic_frames['damaged_roads'] = frame_to_process

            # Save the last problematic frame for each category
            for category, frame_data in problematic_frames.items():
                if frame_data is not None:
                    save_problematic_frame(frame_data, camera_id, category)

            # Clear the batch for the next interval
            frames_batch = []
            last_processing_time = current_time
        
        time.sleep(1) # Small delay to prevent high CPU usage

    cap.release()
    print(f"🛑 Stream processing stopped for Camera ID: {camera_id}.")

def save_problematic_frame(frame, camera_id, category):
    """Saves the detected frame to the correct folder, overwriting the previous one."""
    try:
        # Create folder if it doesn't exist
        output_dir = os.path.join('static', category)
        os.makedirs(output_dir, exist_ok=True)
        
        # Clear any old images in the folder to store only the latest one
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))

        # Generate filename with metadata
        now = datetime.now()
        timestamp = now.strftime("%H%M%S")
        datestamp = now.strftime("%d%m%y")
        filename = f"{camera_id}_{timestamp}_{datestamp}.jpg"
        
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"✅ Saved detection for '{category}' to {filepath}")
    except Exception as e:
        print(f"❌ Error saving frame for {category}: {e}")

# --- Flask API Endpoints ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_stream():
    """Starts the background video processing thread."""
    global processing_thread, stop_processing_flag
    
    data = request.json
    rtsp_url = data.get('rtsp_url')
    camera_id = data.get('camera_id')

    if not rtsp_url or not camera_id:
        return jsonify({'status': 'error', 'message': 'RTSP URL and Camera ID are required.'}), 400

    if processing_thread and processing_thread.is_alive():
        return jsonify({'status': 'error', 'message': 'A stream is already being processed. Stop it first.'}), 400

    # Reset the stop flag and start a new thread
    stop_processing_flag.clear()
    processing_thread = threading.Thread(target=process_rtsp_stream, args=(rtsp_url, camera_id))
    processing_thread.daemon = True # Allows main thread to exit even if this thread is running
    processing_thread.start()
    
    return jsonify({'status': 'success', 'message': f'Started monitoring stream for Camera ID: {camera_id}'})

@app.route('/stop', methods=['POST'])
def stop_stream():
    """Stops the background video processing thread."""
    global processing_thread, stop_processing_flag
    
    if processing_thread and processing_thread.is_alive():
        stop_processing_flag.set()
        processing_thread.join(timeout=5) # Wait for the thread to finish
        processing_thread = None
        return jsonify({'status': 'success', 'message': 'Stream processing stopped.'})
    
    return jsonify({'status': 'error', 'message': 'No active stream to stop.'})

@app.route('/latest_detections')
def get_latest_detections():
    """API endpoint for the frontend to fetch the latest detection info."""
    detections = {}
    categories = ['garbage_overflow', 'stray_animals', 'damaged_roads']
    
    for category in categories:
        detections[category] = None
        folder_path = os.path.join('static', category)
        try:
            if os.path.exists(folder_path) and os.listdir(folder_path):
                # Get the single image file in the directory
                filename = os.listdir(folder_path)[0]
                
                # Parse filename: (camera_id)_timestamp(hhmmss)_date(ddmmyy).jpg
                parts = filename.replace('.jpg', '').split('_')
                cam_id = parts[0]
                time_str = parts[1] # hhmmss
                date_str = parts[2] # ddmmyy
                
                # Format for display
                formatted_time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
                formatted_date = f"{date_str[:2]}-{date_str[2:4]}-20{date_str[4:]}"
                
                detections[category] = {
                    'image_url': f'/{folder_path}/{filename}',
                    'camera_id': cam_id,
                    'time': formatted_time,
                    'date': formatted_date
                }
        except Exception as e:
            print(f"Could not parse file in {category}: {e}")

    return jsonify(detections)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
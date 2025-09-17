# ==============================================================================
# Hybrid RTSP & Video File Monitoring and Detection System
# ==============================================================================
# To Run:
# 1. Activate your conda environment: `conda activate detection_env`
# 2. Run the script: `python app.py`
# ==============================================================================

import os
import cv2
import time
import threading
import csv
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from werkzeug.utils import secure_filename

# --- Configuration ---
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Suppress console output for GET/POST requests
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- Global State ---
processing_thread = None
stop_processing_flag = threading.Event()

# ==============================================================================
# --- PyTorch Model & Location Data Setup ---
# ==============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {'garbage_overflow': None, 'stray_animals': None, 'damaged_roads': None}
CLASS_NAMES = {
    'garbage_overflow': ["negative", "positive"],
    'stray_animals': ["negative", "positive"],
    'damaged_roads': ["negative", "positive"]
}
TRANSFORMS = None
LOCATION_DATA = {}

def load_location_data(file_path='loc.csv'):
    global LOCATION_DATA
    try:
        with open(file_path, mode='r') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                LOCATION_DATA[row['CamID']] = {'lat': row['Latitude'], 'lon': row['Longitude']}
        print(f"✅ Successfully loaded location data for {len(LOCATION_DATA)} cameras.")
    except FileNotFoundError:
        print(f"⚠️ Warning: {file_path} not found. Location data will not be available.")
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")

def load_pytorch_model(model_path, num_classes=2):
    try:
        model = resnet50()
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model = model.to(DEVICE)
        model.eval()
        print(f"✅ Successfully loaded model from: {model_path}")
        return model
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"❌ An error occurred while loading model {model_path}: {e}")
        return None

def setup_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
# ADD THIS NEW ENDPOINT to app.py

@app.route('/status')
def get_status():
    """Checks if a background process is currently running."""
    if processing_thread and processing_thread.is_alive():
        return jsonify({'status': 'processing'})
    return jsonify({'status': 'idle'})

# NEW, CORRECTED VERSION
def classify_frame(frame, model, transform, class_names):
    """Classifies a single video frame (NumPy array) using a loaded PyTorch model."""
    if model is None:
        return False

    try:
        # The line below is the corrected one (cv2 instead of cv)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = probabilities.argmax(1).item()
            predicted_class = class_names[predicted_idx]
        return predicted_class == "positive"
    except Exception as e:
        print(f"❌ Error during frame classification: {e}")
        return False

# ==============================================================================
# --- Detection Functions (Used by both RTSP and Video processing) ---
# ==============================================================================

def detect_garbage_overflow(frame):
    return classify_frame(frame, MODELS['garbage_overflow'], TRANSFORMS, CLASS_NAMES['garbage_overflow'])

def detect_stray_animals(frame):
    return classify_frame(frame, MODELS['stray_animals'], TRANSFORMS, CLASS_NAMES['stray_animals'])

def detect_damaged_roads(frame):
    return classify_frame(frame, MODELS['damaged_roads'], TRANSFORMS, CLASS_NAMES['damaged_roads'])

# ==============================================================================
# --- Core Application Logic (RTSP Stream and Local Video) ---
# ==============================================================================

def save_problematic_frame(frame, camera_id, category, timestamp, datestamp):
    try:
        output_dir = os.path.join('static', category)
        os.makedirs(output_dir, exist_ok=True)
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))
        filename = f"{camera_id}_{timestamp}_{datestamp}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"✅ Saved detection for '{category}' to {filepath}")
    except Exception as e:
        print(f"❌ Error saving frame for {category}: {e}")

def process_rtsp_stream(rtsp_url, camera_id):
    print(f"🚀 Starting RTSP stream processing for Camera ID: {camera_id}...")
    # ... [The RTSP processing logic remains the same as before, but calls the new save function]
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"❌ Error: Could not open RTSP stream at {rtsp_url}")
        return

    frames_batch = []
    last_capture_time = time.time()
    last_processing_time = time.time()

    while not stop_processing_flag.is_set():
        current_time = time.time()
        if current_time - last_capture_time >= 5:
            ret, frame = cap.read()
            if not ret:
                print("❌ Warning: Failed to grab frame. Reconnecting...")
                time.sleep(2)
                cap.release(); cap = cv2.VideoCapture(rtsp_url)
                continue
            frames_batch.append(frame)
            last_capture_time = current_time

        if current_time - last_processing_time >= 30 and frames_batch:
            print(f"\n⏳ Processing a batch of {len(frames_batch)} frames...")
            problematic_frames = {'garbage_overflow': None, 'stray_animals': None, 'damaged_roads': None}
            for frame_to_process in frames_batch:
                if detect_garbage_overflow(frame_to_process): problematic_frames['garbage_overflow'] = frame_to_process
                if detect_stray_animals(frame_to_process): problematic_frames['stray_animals'] = frame_to_process
                if detect_damaged_roads(frame_to_process): problematic_frames['damaged_roads'] = frame_to_process
            
            now = datetime.now()
            timestamp = now.strftime("%H%M%S")
            datestamp = now.strftime("%d%m%y")
            for category, frame_data in problematic_frames.items():
                if frame_data is not None:
                    save_problematic_frame(frame_data, camera_id, category, timestamp, datestamp)
            
            frames_batch = []
            last_processing_time = current_time
        
        time.sleep(1)
    cap.release()
    print(f"🛑 RTSP stream processing stopped for Camera ID: {camera_id}.")


def process_local_video(video_path):
    """Processes a local video file, saving the last problematic frame if any detection occurs."""
    print(f"🚀 Starting local video processing for: {video_path}...")
    try:
        filename = os.path.basename(video_path)
        parts = filename.replace('.mp4', '').split('_')
        camera_id, timestamp, datestamp = parts[0], parts[1], parts[2]
    except Exception:
        print(f"⚠️ Warning: Could not parse metadata from filename '{filename}'. Using generic info.")
        camera_id, timestamp, datestamp = "VID01", "000000", "010124"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {video_path}")
        return

    last_problematic_frames = {'garbage_overflow': None, 'stray_animals': None, 'damaged_roads': None}
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # Process every Nth frame to speed things up (e.g., every 15th frame for a 30fps video is twice a sec)
        if frame_count % 15 == 0:
            if detect_garbage_overflow(frame): last_problematic_frames['garbage_overflow'] = frame.copy()
            if detect_stray_animals(frame): last_problematic_frames['stray_animals'] = frame.copy()
            if detect_damaged_roads(frame): last_problematic_frames['damaged_roads'] = frame.copy()
    
    cap.release()

    # After processing the whole video, save the last detected problematic frame for each category
    for category, frame_data in last_problematic_frames.items():
        if frame_data is not None:
            save_problematic_frame(frame_data, camera_id, category, timestamp, datestamp)
    
    # Clean up the uploaded file
    try:
        os.remove(video_path)
    except Exception as e:
        print(f"❌ Error removing temporary video file: {e}")

    print(f"✅ Finished processing video {filename}.")

# ==============================================================================
# --- Flask API Endpoints ---
# ==============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_rtsp', methods=['POST'])
def start_rtsp():
    global processing_thread, stop_processing_flag
    data = request.json
    rtsp_url, camera_id = data.get('rtsp_url'), data.get('camera_id')
    if not rtsp_url or not camera_id: return jsonify({'status': 'error', 'message': 'RTSP URL and Camera ID are required.'}), 400
    if processing_thread and processing_thread.is_alive(): return jsonify({'status': 'error', 'message': 'A process is already running.'}), 400
    stop_processing_flag.clear()
    processing_thread = threading.Thread(target=process_rtsp_stream, args=(rtsp_url, camera_id))
    processing_thread.daemon = True
    processing_thread.start()
    return jsonify({'status': 'success', 'message': f'Started monitoring RTSP stream for Camera ID: {camera_id}'})

@app.route('/process_video', methods=['POST'])
def process_video_endpoint():
    global processing_thread
    if 'video' not in request.files: return jsonify({'status': 'error', 'message': 'No video file part'}), 400
    file = request.files['video']
    if file.filename == '': return jsonify({'status': 'error', 'message': 'No selected file'}), 400
    
    if file:
        if processing_thread and processing_thread.is_alive(): return jsonify({'status': 'error', 'message': 'A process is already running.'}), 400
        
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        processing_thread = threading.Thread(target=process_local_video, args=(video_path,))
        processing_thread.daemon = True
        processing_thread.start()
        return jsonify({'status': 'success', 'message': f'Started processing video: {filename}'})
    return jsonify({'status': 'error', 'message': 'File upload failed.'})

@app.route('/stop', methods=['POST'])
def stop_stream():
    global processing_thread, stop_processing_flag
    if processing_thread and processing_thread.is_alive():
        stop_processing_flag.set()
        processing_thread.join(timeout=5)
        processing_thread = None
        return jsonify({'status': 'success', 'message': 'Process stopped.'})
    return jsonify({'status': 'error', 'message': 'No active process to stop.'})

@app.route('/latest_detections')
def get_latest_detections():
    detections = {}
    categories = ['garbage_overflow', 'stray_animals', 'damaged_roads']
    for category in categories:
        detections[category] = None
        folder_path = os.path.join('static', category)
        try:
            if os.path.exists(folder_path) and os.listdir(folder_path):
                filename = os.listdir(folder_path)[0]
                parts = filename.replace('.jpg', '').split('_')
                cam_id, time_str, date_str = parts[0], parts[1], parts[2]
                location = LOCATION_DATA.get(cam_id, {'lat': 'N/A', 'lon': 'N/A'})
                formatted_time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
                formatted_date = f"{date_str[:2]}-{date_str[2:4]}-20{date_str[4:]}"
                detections[category] = {
                    'image_url': f'/{folder_path}/{filename}', 'camera_id': cam_id,
                    'time': formatted_time, 'date': formatted_date,
                    'latitude': location['lat'], 'longitude': location['lon']
                }
        except Exception as e:
            print(f"Could not parse file in {category}: {e}")
    return jsonify(detections)

# ==============================================================================
# --- Main Execution ---
# ==============================================================================
if __name__ == '__main__':
    print("-----------------------------------------------------")
    load_location_data()
    print("🧠 Initializing all models, please wait...")
    TRANSFORMS = setup_transforms()
    model_files = {
        'damaged_roads': 'roaddamage.pth',
        'garbage_overflow': 'garbage_model.pth',
        'stray_animals': 'animal_model.pth'
    }
    for model_name, file_name in model_files.items():
        model_path = os.path.join('models', file_name)
        MODELS[model_name] = load_pytorch_model(model_path, num_classes=2)
    
    print("-----------------------------------------------------")
    print("✅ All models initialized. Starting Flask server...")
    print(f"➡️  Open this link in your browser: http://127.0.0.1:5000")
    print("-----------------------------------------------------")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
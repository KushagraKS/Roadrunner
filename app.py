import os
import cv2
import time
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify, request

# --- NEW: PyTorch Model Integration Imports ---
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np

# --- Configuration ---
app = Flask(__name__)
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- Global State ---
processing_thread = None
stop_processing_flag = threading.Event()

# ==============================================================================
# --- NEW: PYTORCH MODEL SETUP & INFERENCE ---
# ==============================================================================

# --- Model Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {
    'garbage_overflow': None,
    'stray_animals': None,
    'damaged_roads': None
}
# Define class names for each model. Assumes "positive" means a detection.
CLASS_NAMES = {
    'damaged_roads': ["negative", "positive"]
}
TRANSFORMS = None

def load_pytorch_model(model_path, num_classes=2):
    """
    Loads a trained ResNet50 model from a .pth file.
    (Adapted from your script)
    """
    try:
        model = resnet50()
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, num_classes)
        )
        # Load the weights from the provided path
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval() # Set model to evaluation mode
        print(f"✅ Successfully loaded model from: {model_path}")
        return model
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"❌ An error occurred while loading model {model_path}: {e}")
        return None

def setup_transforms():
    """
    Defines the image transformations required by the model.
    (Adapted from your script)
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def classify_frame(frame, model, transform, class_names):
    """
    Classifies a single video frame (NumPy array) using a loaded PyTorch model.
    This is the key function that bridges OpenCV and PyTorch.
    """
    if model is None:
        return False # Model not loaded, cannot classify

    try:
        # 1. Convert OpenCV frame (BGR) to PIL Image (RGB)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 2. Apply transformations and add batch dimension
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # 3. Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = probabilities.argmax(1).item()
            predicted_class = class_names[predicted_idx]

        # 4. Return True if the prediction is "positive"
        return predicted_class == "positive"

    except Exception as e:
        print(f"❌ Error during frame classification: {e}")
        return False

# ==============================================================================
# --- ORIGINAL FLASK APPLICATION CODE (with modifications) ---
# ==============================================================================

# --- Detection Functions (Now using real models) ---

def detect_garbage_overflow(frame):
    """Placeholder for garbage model."""
    # TODO: Integrate your garbage model here using the classify_frame function
    # return classify_frame(frame, MODELS['garbage_overflow'], TRANSFORMS, CLASS_NAMES['garbage_overflow'])
    return False # Returning False until model is integrated

def detect_stray_animals(frame):
    """Placeholder for animal model."""
    # TODO: Integrate your stray animal model here
    # return classify_frame(frame, MODELS['stray_animals'], TRANSFORMS, CLASS_NAMES['stray_animals'])
    return False # Returning False until model is integrated

def detect_damaged_roads(frame):
    """INTEGRATED: Uses the loaded PyTorch model for road damage detection."""
    return classify_frame(frame, MODELS['damaged_roads'], TRANSFORMS, CLASS_NAMES['damaged_roads'])

# --- Core Video Processing Logic (No changes needed here) ---

def process_rtsp_stream(rtsp_url, camera_id):
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
        if current_time - last_capture_time >= 5:
            ret, frame = cap.read()
            if not ret:
                print("❌ Warning: Failed to grab frame. Stream may have ended.")
                time.sleep(2)
                cap.release()
                cap = cv2.VideoCapture(rtsp_url)
                continue
            frames_batch.append(frame)
            last_capture_time = current_time

        if current_time - last_processing_time >= 30 and frames_batch:
            print(f"\n⏳ Processing a batch of {len(frames_batch)} frames...")
            problematic_frames = {
                'garbage_overflow': None, 'stray_animals': None, 'damaged_roads': None
            }
            for frame_to_process in frames_batch:
                if detect_garbage_overflow(frame_to_process):
                    problematic_frames['garbage_overflow'] = frame_to_process
                if detect_stray_animals(frame_to_process):
                    problematic_frames['stray_animals'] = frame_to_process
                if detect_damaged_roads(frame_to_process):
                    problematic_frames['damaged_roads'] = frame_to_process
            
            for category, frame_data in problematic_frames.items():
                if frame_data is not None:
                    save_problematic_frame(frame_data, camera_id, category)
            
            frames_batch = []
            last_processing_time = current_time
        
        time.sleep(1)

    cap.release()
    print(f"🛑 Stream processing stopped for Camera ID: {camera_id}.")

def save_problematic_frame(frame, camera_id, category):
    try:
        output_dir = os.path.join('static', category)
        os.makedirs(output_dir, exist_ok=True)
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))
        now = datetime.now()
        timestamp = now.strftime("%H%M%S")
        datestamp = now.strftime("%d%m%y")
        filename = f"{camera_id}_{timestamp}_{datestamp}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"✅ Saved detection for '{category}' to {filepath}")
    except Exception as e:
        print(f"❌ Error saving frame for {category}: {e}")

# --- Flask API Endpoints (No changes needed here) ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_stream():
    global processing_thread, stop_processing_flag
    data = request.json
    rtsp_url = data.get('rtsp_url')
    camera_id = data.get('camera_id')
    if not rtsp_url or not camera_id:
        return jsonify({'status': 'error', 'message': 'RTSP URL and Camera ID are required.'}), 400
    if processing_thread and processing_thread.is_alive():
        return jsonify({'status': 'error', 'message': 'A stream is already being processed.'}), 400
    stop_processing_flag.clear()
    processing_thread = threading.Thread(target=process_rtsp_stream, args=(rtsp_url, camera_id))
    processing_thread.daemon = True
    processing_thread.start()
    return jsonify({'status': 'success', 'message': f'Started monitoring stream for Camera ID: {camera_id}'})

@app.route('/stop', methods=['POST'])
def stop_stream():
    global processing_thread, stop_processing_flag
    if processing_thread and processing_thread.is_alive():
        stop_processing_flag.set()
        processing_thread.join(timeout=5)
        processing_thread = None
        return jsonify({'status': 'success', 'message': 'Stream processing stopped.'})
    return jsonify({'status': 'error', 'message': 'No active stream to stop.'})

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
                formatted_time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
                formatted_date = f"{date_str[:2]}-{date_str[2:4]}-20{date_str[4:]}"
                detections[category] = {
                    'image_url': f'/{folder_path}/{filename}',
                    'camera_id': cam_id, 'time': formatted_time, 'date': formatted_date
                }
        except Exception as e:
            print(f"Could not parse file in {category}: {e}")
    return jsonify(detections)

# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================
if __name__ == '__main__':
    # --- MODIFIED: Load models and transforms at startup ---
    print("-----------------------------------------------------")
    print("🧠 Initializing models, please wait...")
    
    TRANSFORMS = setup_transforms()
    
    # Load Road Damage Model
    road_model_path = os.path.join('models', 'roaddamage.pth')
    MODELS['damaged_roads'] = load_pytorch_model(road_model_path, num_classes=2)
    
    # TODO: Load your other models here when ready
    # garbage_model_path = os.path.join('models', 'garbage_model.pth')
    # MODELS['garbage_overflow'] = load_pytorch_model(garbage_model_path, num_classes=2)
    
    # animal_model_path = os.path.join('models', 'animal_model.pth')
    # MODELS['stray_animals'] = load_pytorch_model(animal_model_path, num_classes=2)
    
    print("✅ Models initialized. Starting Flask server...")
    print("-----------------------------------------------------")
    
    app.run(host='0.0.0.t', port=5000, debug=False, use_reloader=False)
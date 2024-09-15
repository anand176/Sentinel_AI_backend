import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip
from threading import Thread, Lock
import requests
import logging
from flask_cors import CORS
 
# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FRAMES_FOLDER'] = 'processed_frames'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}
 
# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FRAMES_FOLDER'], exist_ok=True)
os.makedirs('anomalous_clips', exist_ok=True)  # Folder for video clips
 
# Load environment variables
load_dotenv()
 
# Load autoencoder model
model = tf.keras.models.load_model("autoencoder_video_complex.h5")
 
# Global variable to store processing status and lock for thread safety
processing_status = {'progress': 0, 'status': 'Starting'}
status_lock = Lock()
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
 
def preprocess_frame(frame):
    frame = cv2.resize(frame, (128, 128))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame.astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=-1)
    frame = np.expand_dims(frame, axis=0)
    return frame
 
def detect_anomaly(autoencoder, frame):
    reconstructed = autoencoder.predict(frame)
    mse = np.mean(np.power(frame - reconstructed, 2))
    threshold = 0.0235
    return mse > threshold
 
def save_video_clip(video_path, start_frame, end_frame, output_clip_path):
    video = VideoFileClip(video_path)
    start_time = start_frame / video.fps
    end_time = end_frame / video.fps
    clip = video.subclip(start_time, end_time)
    clip.write_videofile(output_clip_path, codec='libx264')
 
# Function to post video clip to another Flask endpoint
def post_video_clip_to_endpoint(video_clip_path):
    endpoint_url = "http://127.0.0.1:5001/upload_video"  # Modify to your target endpoint
    try:
        with open(video_clip_path, 'rb') as video_file:
            files = {'file': (os.path.basename(video_clip_path), video_file)}
            response = requests.post(endpoint_url, files=files)
            if response.status_code == 200:
                print("Video clip posted successfully")
            else:
                print(f"Failed to post video clip, status code: {response.status_code}")
    except Exception as e:
        logging.error(f"Error posting video clip to endpoint: {e}")
 
def process_video(video_path, output_dir_clip):
    global processing_status
    try:
        cap = cv2.VideoCapture(video_path)
        i = 0
        warm_up_frames = 60
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
        while True:
            ret, frame = cap.read()
            if not ret:
                break
 
            preprocessed_frame = preprocess_frame(frame)
 
            if i > warm_up_frames:
                if detect_anomaly(model, preprocessed_frame):
                    # Save video clip of the anomaly
                    start_frame = max(0, i - 25)
                    end_frame = min(total_frames, i + 600)
                    clip_path = os.path.join(output_dir_clip, f"anomalous_clip_0.mp4")
                    save_video_clip(video_path, start_frame, end_frame, clip_path)
 
                    # Post the saved video clip to another endpoint
                    post_video_clip_to_endpoint(clip_path)
 
                    with status_lock:
                        processing_status['progress'] = 100
                        processing_status['status'] = 'Completed'
                    break
            else:
                with status_lock:
                    processing_status['progress'] = int((i / total_frames) * 100)
                    processing_status['status'] = f'Processing frame {i} of {total_frames}'
 
            i += 1
 
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        logging.error(f"Error occurred during video processing: {e}")
        with status_lock:
            processing_status['progress'] = 0
            processing_status['status'] = 'Error occurred'
 
@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        output_dir_clip = 'anomalous_clips'
        # Start processing in a separate thread
        thread = Thread(target=process_video, args=(video_path, output_dir_clip))
        thread.start()
        return jsonify({'message': 'File uploaded successfully, processing started'}), 200
    return jsonify({'error': 'Invalid file type'}), 400
 
@app.route('/progress', methods=['GET'])
def get_progress():
    try:
        with status_lock:
            return jsonify(processing_status)
    except Exception as e:
        logging.error(f"Error retrieving progress: {e}")
        return jsonify({"progress": 0, "status": "Error"}), 500
 
# Endpoint to receive video clip posted by the process_video function
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
 
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)
        return jsonify({'message': 'Video uploaded successfully', 'file_path': file_path}), 200
 
if __name__ == '__main__':
    app.run(debug=True)
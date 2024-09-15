import os
import google.generativeai as genai
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import requests  # Import requests to handle HTTP POST request

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'anomalous_clips1'  # Folder to store videos

# Load environment variables
load_dotenv()
genai.configure(api_key='api_key')

# Create the 'anomalous_clips1' folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

narration_store = {"narration": "", "video": ""}  # Store both video and narration

# Function to generate narration using Gemini
def get_gemini_video_narration(video_path):
    video_file = genai.upload_file(path=video_path)
    prompt = "Describe the possible anomaly in this video in single sentence"
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
    response = model.generate_content([prompt, video_file], request_options={"timeout": 600})
    genai.delete_file(video_file.name)
    return response.text

# Route to upload video, save it in 'anomalous_clips1', and generate narration
@app.route('/upload_video', methods=['POST'])
def upload_video():
    print("Upload video endpoint called")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the video file in 'anomalous_clips1'
    if file:
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        # Generate narration using Gemini
        narration = get_gemini_video_narration(video_path)
        narration_store["narration"] = narration
        narration_store["video"] = filename

        print(f"Video uploaded: {filename}, Narration generated: {narration}")

        # Post to the `/narration` endpoint
        try:
            response = requests.post('http://127.0.0.1:5001/narration_post', json={
                "narration": narration,
                "video": filename
            })
            if response.status_code == 200:
                print("Narration and video successfully posted to narration endpoint")
            else:
                print(f"Failed to post narration and video: {response.text}")
        except Exception as e:
            print(f"Error posting to narration endpoint: {e}")

        return jsonify({"message": "Video uploaded and narration generated"}), 200

    return jsonify({"error": "Invalid file"}), 400

# Route to receive and store narration and video from POST request
@app.route('/narration_post', methods=['POST'])
def narration_post():
    print("Narration post endpoint called")
    data = request.get_json()
    if "narration" in data and "video" in data:
        narration_store["narration"] = data["narration"]
        narration_store["video"] = data["video"]
        print(f"Narration and video received: {data['narration']}, {data['video']}")
        return jsonify({"message": "Narration and video stored successfully"}), 200
    return jsonify({"error": "Invalid data"}), 400

# Route to get the video and generated narration
@app.route('/narration', methods=['GET'])
def get_narration():
    print("Narration endpoint called")
    video = narration_store.get("video", "")
    narration = narration_store.get("narration", "No narration available")
    if video:
        print(f"Serving narration: {narration}, Video: {video}")
        return jsonify({"narration": narration, "video": video})
    return jsonify({"error": "No video available"}), 400

# Route to serve the video file
@app.route('/anomalous_clips1/<filename>', methods=['GET'])
def serve_video(filename):
    print(f"Serving video file: {filename}")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("Flask app started")
    app.run(port=5001, debug=True)

import cv2
import mediapipe as mp
import numpy as np
import joblib
import pickle
import pandas as pd
import wave
import json
from vosk import Model, KaldiRecognizer
import io
import os
from pydub import AudioSegment
from flask import Flask, render_template_string, request, jsonify, Response, send_file
import threading
import time
import queue
from collections import defaultdict, deque
import base64
import webbrowser
import socket
import random
import qrcode
import sys
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter app

# Global variables for mode management
current_mode = 1
asl_text = ""
voice_results = {}

# Mode 1: ASL Alphabet Prediction
class ASLPredictor:
    def __init__(self):
        self.model_file = "asl-alphabets/asl_model.pkl"
        self.clf = None
        self.class_names = None
        self.mp_hands = mp.solutions.hands
        self.hands = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.prediction_history = deque(maxlen=90)  # 3 seconds at 30fps
        self.stable_prediction = None
        self.stable_start_time = None
        self.current_text = ""
        
    def initialize(self):
        try:
            self.clf, self.class_names = joblib.load(self.model_file)
            self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
            return True
        except FileNotFoundError:
            print("ASL model file not found")
            return False
    
    def normalize_landmarks(self, landmarks):
        landmarks = landmarks.reshape(21, 3)
        wrist = landmarks[0]
        landmarks -= wrist
        return landmarks.flatten()
    
    def extract_landmarks(self, hand_landmarks):
        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        return np.array(coords, dtype=np.float32).reshape(1, -1)
    
    def predict_frame(self, frame):
        if self.hands is None:
            return None, None
            
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        
        prediction = None
        confidence = 0
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                landmarks = self.extract_landmarks(hand_landmarks)
                landmarks = self.normalize_landmarks(landmarks.flatten())
                pred_id = self.clf.predict(landmarks.reshape(1, -1))[0]
                pred_class = self.class_names[pred_id]
                confidence = np.max(self.clf.predict_proba(landmarks.reshape(1, -1)))
                
                # Track prediction stability
                current_time = time.time()
                self.prediction_history.append((pred_class, current_time))
                
                # Check if we have a stable prediction for 3+ seconds
                if len(self.prediction_history) >= 90:  # 3 seconds at 30fps
                    recent_predictions = [p[0] for p in list(self.prediction_history)[-90:]]
                    if all(p == recent_predictions[0] for p in recent_predictions):
                        if self.stable_prediction != recent_predictions[0]:
                            self.stable_prediction = recent_predictions[0]
                            self.stable_start_time = current_time
                        elif current_time - self.stable_start_time >= 3.0:
                            prediction = self.stable_prediction
                            self.stable_prediction = None
                            self.prediction_history.clear()
                    else:
                        self.stable_prediction = None
                        self.stable_start_time = None
                
                cv2.putText(frame, f"Prediction: {pred_class} ({confidence:.2f})",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.stable_prediction = None
            self.stable_start_time = None
        
        return frame, prediction

# Mode 2: Voice Translation
class VoiceTranslator:
    def __init__(self):
        self.models = [
            r"vosk/vosk-model-small-en-us-0.15",
            r"vosk/vosk-model-en-us-0.42-gigaspeech",
            r"vosk/vosk-model-en-us-0.22"
        ]
        self.asl_words = []
        self.embed_model = None
        self.nn = None
        self.tamil_words = None
        self.df = None
        self.video_dir = "videos"
        
    def initialize(self):
        try:
            # Load ASL words
            with open("vosk/asl_words.txt", "r", encoding="utf-8") as f:
                self.asl_words = [w.strip() for w in f.readlines() if w.strip()]
            
            # Load translation models
            self.embed_model = pickle.load(open("vosk/models/embed_model.pkl", "rb"))
            self.nn = pickle.load(open("vosk/models/eng_tamil_nn.pkl", "rb"))
            self.tamil_words = pickle.load(open("vosk/models/tamil_words.pkl", "rb"))
            
            # Load dataset
            self.df = pd.read_csv("vosk/eng_tamil_dataset.csv")
            
            return True
        except Exception as e:
            print(f"Error initializing voice translator: {e}")
            return False
    
    def transcribe_audio(self, audio_data):
        for model_path in self.models:
            if not os.path.exists(model_path):
                continue
                
            try:
                model = Model(model_path)
                
                # Convert audio to WAV format
                audio = AudioSegment.from_file(io.BytesIO(audio_data))
                audio = audio.set_channels(1)  # Mono
                audio = audio.set_frame_rate(16000)  # 16kHz
                wav_data = io.BytesIO()
                audio.export(wav_data, format='wav')
                wav_data.seek(0)
                wf = wave.open(wav_data, 'rb')
                
                rec = KaldiRecognizer(model, wf.getframerate())
                transcription = ""
                
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        transcription += result.get('text', '') + ' '
                
                final_result = json.loads(rec.FinalResult())
                transcription += final_result.get('text', '')
                
                if transcription.strip():
                    return transcription.strip().lower()
                    
            except Exception as e:
                print(f"Error with model {model_path}: {e}")
                continue
        
        return ""
    
    def translate_to_tamil(self, english_word):
        if english_word not in self.asl_words:
            return None, None, None
        
        try:
            emb = self.embed_model.encode([english_word])
            dist, idx = self.nn.kneighbors(emb)
            tamil_word = self.tamil_words[idx[0][0]]
            
            # Find video file
            video_path = os.path.join(self.video_dir, f"{english_word}.mp4")
            if os.path.exists(video_path):
                return english_word, tamil_word, video_path
            else:
                return english_word, tamil_word, None
                
        except Exception as e:
            print(f"Translation error: {e}")
            return english_word, None, None

# Initialize predictors
asl_predictor = ASLPredictor()
voice_translator = VoiceTranslator()

asl_predictor.initialize()
voice_translator.initialize()

# Flask routes for Flutter app
@app.route('/')
def index():
    return jsonify({"status": "ASL & Voice Translator Server Running"})

@app.route('/switch_mode', methods=['POST'])
def switch_mode():
    global current_mode
    data = request.get_json()
    current_mode = data.get('mode', 1)
    return jsonify({'status': 'success', 'mode': current_mode})

@app.route('/video_feed')
def video_feed():
    def generate():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return
        
        while current_mode == 1:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame, prediction = asl_predictor.predict_frame(frame)
            
            if prediction:
                global asl_text
                asl_text += prediction
            
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        
        cap.release()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_text')
def get_text():
    return jsonify({'text': asl_text})

@app.route('/add_space', methods=['POST'])
def add_space():
    global asl_text
    asl_text += ' '
    return jsonify({'text': asl_text})

@app.route('/delete_char', methods=['POST'])
def delete_char():
    global asl_text
    if asl_text:
        asl_text = asl_text[:-1]
    return jsonify({'text': asl_text})

@app.route('/clear_text', methods=['POST'])
def clear_text():
    global asl_text
    asl_text = ''
    return jsonify({'text': asl_text})

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'})
    
    audio_file = request.files['audio']
    audio_data = audio_file.read()
    
    # Transcribe audio
    transcription = voice_translator.transcribe_audio(audio_data)
    
    if not transcription:
        return jsonify({'error': 'No speech detected'})
    
    # Split into words and process each
    words = transcription.split()
    results = []
    
    for word in words:
        english_word, tamil_word, video_path = voice_translator.translate_to_tamil(word)
        if english_word:
            result = {
                'english': english_word,
                'tamil': tamil_word,
                'video_url': f'/video/{english_word}' if video_path else None
            }
            results.append(result)
    
    # Return first valid result
    if results:
        return jsonify(results[0])
    else:
        return jsonify({'error': 'No valid words found'})

@app.route('/video/<word>')
def serve_video(word):
    video_path = os.path.join('videos', f'{word}.mp4')
    if os.path.exists(video_path):
        return send_file(video_path, mimetype='video/mp4')
    else:
        return "Video not found", 404

def listen_for_quit():
    """Listen for 'q' key press to show QR code"""
    import keyboard
    while True:
        try:
            if keyboard.is_pressed('q'):
                # Generate QR code with local network IP
                local_ip = get_local_ip()
                network_url = f"http://{local_ip}:{port}"
                generate_qr_code(network_url)
                time.sleep(1)  # Prevent multiple triggers
        except:
            pass
        time.sleep(0.1)

def get_local_ip():
    """Get the local IP address for network access"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def generate_qr_code(url):
    """Generate and print QR code for the given URL"""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=1,
        border=1,
    )
    qr.add_data(url)
    qr.make(fit=True)
    
    # Print QR code to terminal
    qr.print_ascii(invert=True)
    print(f"\nScan QR code or visit: {url}")
    print(f"Access from other devices: http://{get_local_ip()}:{port}")
    print(f"Flutter app should connect to: http://{get_local_ip()}:{port}")
    print("=" * 60)

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

if __name__ == '__main__':
    port = find_free_port()
    local_ip = get_local_ip()
    
    print(f"Starting server on port {port}")
    print(f"Local access: http://localhost:{port}")
    print(f"Network access: http://{local_ip}:{port}")
    print(f"Flutter app should connect to: http://{local_ip}:{port}")
    print("Press 'q' in terminal to show QR code")
    
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

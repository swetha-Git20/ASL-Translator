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

# Try to import MediaPipe tasks, fall back to solutions if not available
try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_NEW_API = True
except ImportError:
    MEDIAPIPE_NEW_API = False
    print("Using legacy MediaPipe API")

app = Flask(__name__)

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
        self.landmarker = None
        self.hands = None
        self.prediction_history = deque(maxlen=90)  # 3 seconds at 30fps
        self.stable_prediction = None
        self.stable_start_time = None
        self.current_text = ""
        
    def initialize(self):
        try:
            self.clf, self.class_names = joblib.load(self.model_file)
            
            if MEDIAPIPE_NEW_API:
                # Use new MediaPipe Tasks API
                model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                model_path = "hand_landmarker.task"
                
                if not os.path.exists(model_path):
                    print("Downloading hand landmarker model...")
                    import urllib.request
                    urllib.request.urlretrieve(model_url, model_path)
                
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=1
                )
                self.landmarker = vision.HandLandmarker.create_from_options(options)
            else:
                # Use legacy MediaPipe Solutions API
                self.hands = mp.solutions.hands.Hands(
                    static_image_mode=False, 
                    max_num_hands=1, 
                    min_detection_confidence=0.5
                )
                self.mp_drawing = mp.solutions.drawing_utils
            
            return True
        except FileNotFoundError:
            print("ASL model file not found")
            return False
        except Exception as e:
            print(f"Error initializing hand detector: {e}")
            return False
    
    def normalize_landmarks(self, landmarks):
        landmarks = landmarks.reshape(21, 3)
        wrist = landmarks[0]
        landmarks -= wrist
        return landmarks.flatten()
    
    def extract_landmarks(self, hand_landmarks):
        coords = []
        if MEDIAPIPE_NEW_API:
            for lm in hand_landmarks:
                coords.extend([lm.x, lm.y, lm.z])
        else:
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
        return np.array(coords, dtype=np.float32).reshape(1, -1)
    
    def predict_frame(self, frame):
        frame = cv2.flip(frame, 1)
        prediction = None
        confidence = 0
        
        try:
            if MEDIAPIPE_NEW_API and self.landmarker:
                # Use new MediaPipe Tasks API
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                result = self.landmarker.detect(mp_image)
                
                if result.hand_landmarks:
                    for hand_landmarks in result.hand_landmarks:
                        # Draw landmarks on frame
                        for landmark in hand_landmarks:
                            x = int(landmark.x * frame.shape[1])
                            y = int(landmark.y * frame.shape[0])
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                        
                        # Extract landmarks for prediction
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
                    
            elif not MEDIAPIPE_NEW_API and self.hands:
                # Use legacy MediaPipe Solutions API
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.hands.process(rgb)
                
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                        
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
            else:
                cv2.putText(frame, "Hand detector not initialized", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
        except Exception as e:
            print(f"Error in predict_frame: {e}")
            cv2.putText(frame, f"Error: {str(e)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame, prediction

# Mode 2: Voice Input and Translation
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

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ASL & Voice Translator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .mode-selector {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .mode-btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        .mode-btn.active {
            background-color: #007bff;
            color: white;
        }
        .mode-btn:not(.active) {
            background-color: #e9ecef;
            color: #333;
        }
        .mode-content {
            display: none;
        }
        .mode-content.active {
            display: block;
        }
        .video-container {
            margin-bottom: 20px;
            text-align: center;
        }
        video, img {
            width: 100%;
            max-width: 640px;
            border: 2px solid #ddd;
            border-radius: 5px;
            height: auto;
        }
        .text-output {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            min-height: 100px;
            font-size: 18px;
            border: 1px solid #ddd;
        }
        .controls {
            margin: 10px 0;
        }
        button {
            padding: 8px 16px;
            margin: 5px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        .record-btn {
            background-color: #dc3545;
            color: white;
        }
        .record-btn.recording {
            background-color: #28a745;
        }
        .space-btn {
            background-color: #007bff;
            color: white;
        }
        .delete-btn {
            background-color: #ffc107;
            color: black;
        }
        .clear-btn {
            background-color: #6c757d;
            color: white;
        }
        .camera-btn {
            background-color: #28a745;
            color: white;
        }
        .result-item {
            background-color: #e9ecef;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
        }
        .result-video {
            width: 100%;
            max-width: 400px;
            margin-top: 10px;
        }
        .device-info {
            background-color: #e7f3ff;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            font-size: 14px;
        }
        .permission-info {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            font-size: 14px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>ASL & Voice Translator</h1>
        
        <div class="device-info" id="deviceInfo">
            Detecting device type...
        </div>
        
        <div class="permission-info">
            📱 <strong>Important:</strong> Click "Start Camera" or "Start Recording" buttons to grant camera/microphone permissions. Browser will show permission popup.
        </div>
        
        <div class="mode-selector">
            <button class="mode-btn active" onclick="switchMode(1)">Mode 1: ASL Alphabet</button>
            <button class="mode-btn" onclick="switchMode(2)">Mode 2: Voice Translation</button>
        </div>
        
        <!-- Mode 1: ASL Alphabet Prediction -->
        <div id="mode1" class="mode-content active">
            <h2>ASL Alphabet Prediction</h2>
            <div class="video-container">
                <video id="mobileVideo" style="display: none;" autoplay playsinline></video>
                <img id="desktopVideo" style="display: none;" alt="ASL Video Feed">
                <div id="cameraPlaceholder">Camera will appear here...</div>
            </div>
            <div class="controls">
                <button class="camera-btn" onclick="startCamera()">Start Camera</button>
                <button class="camera-btn" onclick="debugCamera()" style="background-color: #6c757d;">Debug Camera</button>
                <button class="space-btn" onclick="addSpace()">Add Space</button>
                <button class="delete-btn" onclick="deleteLastChar()">Delete Last</button>
                <button class="clear-btn" onclick="clearText()">Clear All</button>
            </div>
            <div class="text-output" id="aslText">Show a sign for 3+ seconds to add it to text...</div>
        </div>
        
        <!-- Mode 2: Voice Translation -->
        <div id="mode2" class="mode-content">
            <h2>Voice Translation</h2>
            <div class="controls">
                <button id="recordBtn" class="record-btn" onclick="toggleRecording()">Start Recording</button>
                <button class="clear-btn" onclick="clearVoiceResults()">Clear Results</button>
            </div>
            <div id="voiceResults"></div>
        </div>
    </div>

    <script>
        let currentMode = 1;
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        let isMobile = false;
        let cameraStream = null;
        let predictionHistory = [];
        let lastPrediction = null;
        let predictionStartTime = null;

        // Detect device type
        function detectDevice() {
            const userAgent = navigator.userAgent.toLowerCase();
            const isMobileDevice = /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(userAgent);
            const isTablet = /ipad|android(?!.*mobile)/i.test(userAgent);
            
            isMobile = isMobileDevice || isTablet;
            
            const deviceInfo = document.getElementById('deviceInfo');
            if (isMobile) {
                deviceInfo.innerHTML = '📱 Mobile device detected - using device camera and microphone';
            } else {
                deviceInfo.innerHTML = '🖥️ Desktop detected - using server camera and microphone';
            }
        }

        function switchMode(mode) {
            currentMode = mode;
            
            // Update button states
            document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.mode-content').forEach(content => content.classList.remove('active'));
            
            if (mode === 1) {
                document.querySelectorAll('.mode-btn')[0].classList.add('active');
                document.getElementById('mode1').classList.add('active');
                stopRecording();
            } else {
                document.querySelectorAll('.mode-btn')[1].classList.add('active');
                document.getElementById('mode2').classList.add('active');
                stopCamera();
            }
            
            // Notify server
            fetch('/switch_mode', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({mode: mode})
            });
        }

        async function startCamera() {
            try {
                // Show loading message
                document.getElementById('cameraPlaceholder').textContent = 'Requesting camera permission...';
                console.log('Requesting camera permission...');
                console.log('Is mobile:', isMobile);
                console.log('User agent:', navigator.userAgent);
                
                // For desktop, use server-side video feed with MediaPipe processing
                if (!isMobile) {
                    document.getElementById('desktopVideo').src = '/video_feed';
                    document.getElementById('desktopVideo').style.display = 'block';
                    document.getElementById('mobileVideo').style.display = 'none';
                    document.getElementById('cameraPlaceholder').style.display = 'none';
                    startDesktopPolling();
                    return;
                }
                
                // For mobile, try different camera constraints
                let constraints = {
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 }
                    }
                };
                
                // For mobile, try rear camera first, then front
                if (isMobile) {
                    constraints.video.facingMode = { ideal: 'user' };
                } else {
                    constraints.video.facingMode = 'user';
                }
                
                console.log('Camera constraints:', constraints);
                
                let stream;
                try {
                    stream = await navigator.mediaDevices.getUserMedia(constraints);
                } catch (mobileError) {
                    console.log('First attempt failed, trying simpler constraints:', mobileError);
                    // Fallback to simpler constraints
                    stream = await navigator.mediaDevices.getUserMedia({ 
                        video: true 
                    });
                }
                
                console.log('Camera stream obtained:', stream);
                
                cameraStream = stream;
                
                // Show device camera feed
                const video = document.getElementById('mobileVideo');
                video.srcObject = stream;
                video.style.display = 'block';
                document.getElementById('desktopVideo').style.display = 'none';
                document.getElementById('cameraPlaceholder').style.display = 'none';
                
                console.log('Camera feed started');
                
                // Start prediction loop
                if (isMobile) {
                    startMobilePrediction();
                } else {
                    startDesktopPolling();
                }
                
            } catch (error) {
                console.error('Error accessing camera:', error);
                document.getElementById('cameraPlaceholder').textContent = 'Camera access denied. Please allow camera permissions and try again.';
                
                // Show specific error messages
                if (error.name === 'NotAllowedError') {
                    alert('Camera permission denied. Please allow camera access in your browser settings and refresh the page.');
                } else if (error.name === 'NotFoundError') {
                    alert('No camera found. Please connect a camera and try again.');
                } else if (error.name === 'NotSupportedError') {
                    alert('Camera not supported by your browser. Please try a different browser like Chrome.');
                } else if (error.name === 'NotReadableError') {
                    alert('Camera is already in use by another application. Please close other apps using the camera.');
                } else {
                    alert('Error accessing camera: ' + error.message + ' (Error: ' + error.name + ')');
                }
            }
        }

        function debugCamera() {
            console.log('=== Camera Debug Info ===');
            console.log('User Agent:', navigator.userAgent);
            console.log('Is Mobile:', isMobile);
            console.log('HTTPS:', window.isSecureContext);
            console.log('MediaDevices supported:', 'mediaDevices' in navigator);
            console.log('GetUserMedia supported:', 'getUserMedia' in navigator.mediaDevices);
            
            // Check available cameras
            if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
                navigator.mediaDevices.enumerateDevices()
                    .then(devices => {
                        console.log('Available devices:', devices);
                        const videoDevices = devices.filter(device => device.kind === 'videoinput');
                        console.log('Video devices:', videoDevices);
                        alert(`Found ${videoDevices.length} camera(s). Check console for details.`);
                    })
                    .catch(err => {
                        console.error('Error enumerating devices:', err);
                        alert('Error checking cameras: ' + err.message);
                    });
            } else {
                alert('Camera enumeration not supported');
            }
        }

        function stopCamera() {
            if (cameraStream) {
                cameraStream.getTracks().forEach(track => track.stop());
                cameraStream = null;
            }
            document.getElementById('mobileVideo').style.display = 'none';
            document.getElementById('desktopVideo').style.display = 'none';
            document.getElementById('cameraPlaceholder').style.display = 'block';
        }

        function startMobilePrediction() {
            // For mobile, we would need to send frames to server for prediction
            // This is a simplified version - in production, you'd use WebRTC
            console.log('Mobile camera started - frame prediction would be implemented here');
        }

        function startDesktopPolling() {
            // Poll server for text updates (no video feed refresh needed)
            setInterval(() => {
                if (currentMode === 1) {
                    fetch('/get_text')
                        .then(response => response.json())
                        .then(data => {
                            if (data.text && data.text !== document.getElementById('aslText').textContent) {
                                document.getElementById('aslText').textContent = data.text;
                            }
                        })
                        .catch(err => console.log('Text poll error:', err));
                }
            }, 1000);
        }

        function addSpace() {
            fetch('/add_space', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    document.getElementById('aslText').textContent = data.text;
                });
        }

        function deleteLastChar() {
            fetch('/delete_char', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    document.getElementById('aslText').textContent = data.text;
                });
        }

        function clearText() {
            fetch('/clear_text', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    document.getElementById('aslText').textContent = data.text;
                });
        }

        function toggleRecording() {
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        }

        async function startRecording() {
            try {
                // Update button to show requesting permission
                document.getElementById('recordBtn').textContent = 'Requesting mic permission...';
                
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    } 
                });
                
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const formData = new FormData();
                    formData.append('audio', audioBlob, 'recording.wav');

                    const response = await fetch('/process_audio', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();
                    displayVoiceResult(result);
                };

                mediaRecorder.start();
                isRecording = true;
                document.getElementById('recordBtn').textContent = 'Stop Recording';
                document.getElementById('recordBtn').classList.add('recording');
            } catch (error) {
                console.error('Error accessing microphone:', error);
                document.getElementById('recordBtn').textContent = 'Start Recording';
                document.getElementById('recordBtn').classList.remove('recording');
                
                // Show specific error messages
                if (error.name === 'NotAllowedError') {
                    alert('Microphone permission denied. Please allow microphone access in your browser settings and refresh the page.');
                } else if (error.name === 'NotFoundError') {
                    alert('No microphone found. Please connect a microphone and try again.');
                } else if (error.name === 'NotSupportedError') {
                    alert('Microphone not supported by your browser. Please try a different browser.');
                } else {
                    alert('Error accessing microphone: ' + error.message);
                }
            }
        }

        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                isRecording = false;
                document.getElementById('recordBtn').textContent = 'Start Recording';
                document.getElementById('recordBtn').classList.remove('recording');
            }
        }

        function displayVoiceResult(result) {
            const resultsDiv = document.getElementById('voiceResults');
            const resultItem = document.createElement('div');
            resultItem.className = 'result-item';
            
            let content = `<strong>English:</strong> ${result.english || 'N/A'}`;
            if (result.tamil) {
                content += `<br><strong>Tamil:</strong> ${result.tamil}`;
            }
            if (result.video_url) {
                content += `<br><video class="result-video" controls><source src="${result.video_url}" type="video/mp4"></video>`;
            } else {
                content += `<br><em>No video available for this word</em>`;
            }
            
            if (result.error) {
                content = `<em>${result.error}</em>`;
            }
            
            resultItem.innerHTML = content;
            resultsDiv.appendChild(resultItem);
        }

        function clearVoiceResults() {
            document.getElementById('voiceResults').innerHTML = '';
        }

        // Initialize on page load
        window.onload = function() {
            detectDevice();
            // Don't auto-start camera - wait for user to click button
            if (!isMobile) {
                document.getElementById('cameraPlaceholder').textContent = 'Click "Start Camera" to begin ASL detection';
            } else {
                document.getElementById('cameraPlaceholder').textContent = 'Click "Start Camera" to use your device camera';
            }
        };
    </script>
</body>
</html>
"""

# Flask routes
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/switch_mode', methods=['POST'])
def switch_mode():
    global current_mode
    data = request.get_json()
    current_mode = data.get('mode', 1)
    return jsonify({'status': 'success', 'mode': current_mode})

@app.route('/video_feed')
def video_feed():
    def generate():
        # Only generate video feed if running on the host machine
        # Mobile devices will use their own camera via WebRTC
        if request.remote_addr in ['127.0.0.1', 'localhost', '::1']:
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
        else:
            # For mobile devices, return a placeholder or error
            yield b'--frame\r\nContent-Type: text/plain\r\n\r\nMobile device should use camera directly\r\n\r\n'
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

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

@app.route('/get_text')
def get_text():
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
    
    # Return the first valid result
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

def get_local_ip():
    """Get the local IP address for network access"""
    try:
        # Create a socket to connect to an external server
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
    print("Press 'q' in terminal to show QR code")
    
    # Start keyboard listener thread
    try:
        keyboard_thread = threading.Thread(target=listen_for_quit, daemon=True)
        keyboard_thread.start()
    except ImportError:
        print("Note: Install 'keyboard' package to enable QR code on 'q' press")
    except Exception as e:
        print(f"Could not start keyboard listener: {e}")
    
    # Open browser automatically with localhost
    webbrowser.open(f'http://localhost:{port}')
    
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

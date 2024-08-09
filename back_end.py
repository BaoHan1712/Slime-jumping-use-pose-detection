import pickle
import time
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from flask import Flask, Response, render_template, jsonify, request

# Load model
model_dict = pickle.load(open('model.pickle', 'rb'))

# Initialize Flask app
app = Flask(__name__)

# Global Variables
counting = 0
weight = 0  # User weight

# MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mouse_down = False

# Labels
labels_dict = {0: 'normal', 1: 'squat'}

def gen_frames():
    global counting, mouse_down

    cap = cv2.VideoCapture(0)
    pTime = 0

    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (480, 320))
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            for landmark in results.pose_landmarks.landmark:
                x = landmark.x
                y = landmark.y

                x_.append(x)
                y_.append(y)

            for landmark in results.pose_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            try:
                prediction = model_dict.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                if predicted_character == 'squat' and not mouse_down:
                    pyautogui.mouseDown()
                    mouse_down = True

                if predicted_character == 'normal' and mouse_down:
                    pyautogui.mouseUp()
                    counting += 1
                    mouse_down = False

            except Exception as e:
                predicted_character = "Unknown pose"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        # cv2.putText(frame, f'Count: {counting}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/info')
def info():
    global counting, weight

    # Calculate calories burned
    if weight > 0:
        calories_burned = (3 * 4.5 * weight) / 200
    else:
        calories_burned = 0
        
    calories_squat = calories_burned *counting

    return jsonify(counting=counting, calories_burned=calories_squat)

@app.route('/set_weight', methods=['POST'])
def set_weight():
    global weight
    weight = float(request.form.get('weight', 0))
    return jsonify(success=True, weight=weight)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

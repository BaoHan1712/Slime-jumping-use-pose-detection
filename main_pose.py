import pickle
import time
import cv2
import mediapipe as mp
import numpy as np
import pyautogui


# Load model
model_dict = pickle.load(open('model.pickle', 'rb'))

cap = cv2.VideoCapture(0)
pTime = 0

# MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mouse_down = False 

# labels
labels_dict = {0: 'normal', 1: 'squat'}


while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        continue

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
                mouse_down = False
                
        except Exception as e:
            predicted_character = "Unknown pose"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

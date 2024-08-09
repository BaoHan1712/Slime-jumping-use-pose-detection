import os
import pickle
import mediapipe as mp
import cv2

# MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.8, min_tracking_confidence=0.8)

DATA_DIR = 'data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = pose.process(img_rgb)
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                x_.append(x)
                y_.append(y)

            for landmark in results.pose_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

import os
import pickle
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

mp_hand_solution = mp.solutions.hands
draw_utils = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

hand_model = mp_hand_solution.Hands(static_image_mode=True, min_detection_confidence=0.3)

SIGN_DATA_DIR = r"C:\Users\aryam\OneDrive\Desktop\sign_language_dataset"
dataset = []
class_labels = []

for category in os.listdir(SIGN_DATA_DIR):
    category_path = os.path.join(SIGN_DATA_DIR, category)

    if os.path.isdir(category_path):
        for image_file in os.listdir(category_path):
            landmarks_data = []
            x_coords = []
            y_coords = []

            image = cv2.imread(os.path.join(category_path, image_file))
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = hand_model.process(image_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        x_coords.append(landmark.x)
                        y_coords.append(landmark.y)

                    for landmark in hand_landmarks.landmark:
                        landmarks_data.append(landmark.x - min(x_coords))
                        landmarks_data.append(landmark.y - min(y_coords))

                dataset.append(landmarks_data)
                class_labels.append(category)

with open('data1.pickle', 'wb') as file:
    pickle.dump({'data': dataset, 'labels': class_labels}, file)
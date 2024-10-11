import pickle
import cv2
import mediapipe as mp
import numpy as np

model_data = pickle.load(open('model1.p', 'rb'))
sign_model = model_data['model']

camera = cv2.VideoCapture(0)

mp_hand_model = mp.solutions.hands
drawing_utils = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

hand_recognition = mp_hand_model.Hands(static_image_mode=True, min_detection_confidence=0.3)

label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I'}

while True:
    frame_data = []
    x_coords = []
    y_coords = []

    ret, video_frame = camera.read()

    frame_height, frame_width, _ = video_frame.shape

    rgb_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)

    hand_results = hand_recognition.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for landmarks in hand_results.multi_hand_landmarks:
            drawing_utils.draw_landmarks(
                video_frame, landmarks, mp_hand_model.HAND_CONNECTIONS, 
                drawing_styles.get_default_hand_landmarks_style(), 
                drawing_styles.get_default_hand_connections_style())

        for landmarks in hand_results.multi_hand_landmarks:
            for i in range(len(landmarks.landmark)):
                x_coords.append(landmarks.landmark[i].x)
                y_coords.append(landmarks.landmark[i].y)

            for i in range(len(landmarks.landmark)):
                frame_data.append(landmarks.landmark[i].x - min(x_coords))
                frame_data.append(landmarks.landmark[i].y - min(y_coords))

        x1 = int(min(x_coords) * frame_width) - 10
        y1 = int(min(y_coords) * frame_height) - 10
        x2 = int(max(x_coords) * frame_width) - 10
        y2 = int(max(y_coords) * frame_height) - 10

        predicted_class = sign_model.predict([np.asarray(frame_data)])
        predicted_sign = label_map[int(predicted_class[0])]

        cv2.rectangle(video_frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(video_frame, predicted_sign, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('video_frame', video_frame)
    cv2.waitKey(1)

camera.release()
cv2.destroyAllWindows()

import os
import cv2

SAVE_DIR = r"C:\Users\aryam\OneDrive\Desktop\sign_language_dataset"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Defining the number of gesture classes and how many samples to collect per class
num_classes = 9
samples_per_class = 100

camera = cv2.VideoCapture(0)

# Looping through each class to collect data
for class_id in range(num_classes):
    class_dir = os.path.join(SAVE_DIR, str(class_id))
    
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for gesture class {class_id}')

    # Wait for user to press "Q" to begin data collection for the class
    while True:
        ret, frame = camera.read()
        cv2.putText(frame, 'Press "Q" to start capturing', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    image_count = 0
    # Capture the specified number of images for each class
    while image_count < samples_per_class:
        ret, frame = camera.read()
        cv2.imshow('Camera Feed', frame)
        cv2.waitKey(1)
        
        # Save each captured frame as an image file
        img_filename = os.path.join(class_dir, f'{image_count}.jpg')
        cv2.imwrite(img_filename, frame)

        image_count += 1

camera.release()
cv2.destroyAllWindows()
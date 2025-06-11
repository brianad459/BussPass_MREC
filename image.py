import numpy as np
from ultralytics import YOLO
import cv2
import os
import kagglehub
import matplotlib.pyplot as plt
import mediapipe as mp

dataset_path = kagglehub.dataset_download("deeppythonist/american-sign-language-dataset")
print(f"Dataset downloaded to: {dataset_path}")

folder_path = os.path.join(dataset_path, 'ASL_Gestures_36_Classes', 'train')

for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(root, filename)
            img = cv2.imread(image_path)


            if img is not None:
                print(f"Loaded: {image_path}")
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title(filename)
                plt.axis('off')
                plt.show()
            else:
                print(f"Could not load: {image_path}")

model = YOLO("yolo11n.pt")

results = model.predict(source=0, show=True)
print(results)


cap = cv2.VideoCapture(1)
currentFrame = 0

if not os.path.exists('data'):
    os.makedirs('data')

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        print("Cannot read image")
        break

    cv2.imshow("frame", frame)
    cv2.imwrite(f"data/frameTesting{currentFrame}.jpg", frame)
    currentFrame += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exit program")
        break

cap.release()
cv2.destroyAllWindows()

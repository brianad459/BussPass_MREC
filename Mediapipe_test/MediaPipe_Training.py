import os
import cv2
import csv
import mediapipe as mp


#This file simply just gathers data
# Gesture detector class
#https://lvimuth.medium.com/hand-detection-in-python-using-opencv-and-mediapipe-30c7b54f5ff4
class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findLandmarkList(self):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[0]
            for lm in myHand.landmark:
                lmlist.append([lm.x, lm.y, lm.z])
        return lmlist

# Setup CSV output
csv_file = "gesture_data_new.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ["label"]
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(header)

# Main program
cap = cv2.VideoCapture(0)
detector = handDetector()

print("Press 'r' for Rock, 'p' for Paper, 's' for Scissor, 'n' for None, 'q' to Quit")

while True:
    success, image = cap.read()
    if not success:
        break

    image = detector.findHands(image)
    lmlist = detector.findLandmarkList()

    key = cv2.waitKey(1) & 0xFF
    label = None

    if key == ord('r'):
        label = 'rock'
    elif key == ord('p'):
        label = 'paper'
    elif key == ord('s'):
        label = 'scissor'

    elif key == ord('t'):
        label = 'stop'
    elif key == ord('g'):
        label = 'game'
    elif key == ord('o'):
        label = "okay"

    elif key == ord('q'):
        break


    # Save to CSV if label and landmarks exist
    if label and lmlist:
        row = [label]
        for lm in lmlist:
            row.extend(lm)
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print(f"Saved: {label}")

    cv2.imshow("Hand Gesture Collection", image)

cap.release()
cv2.destroyAllWindows()

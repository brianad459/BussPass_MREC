import os
import cv2
import csv
import mediapipe as mp

# =========================
# Hand detector class
# =========================
class HandDetector:
    def __init__(self,
                 mode=False,
                 maxHands=2,
                 modelComplexity=1,
                 detectionCon=0.5,
                 trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=self.modelComplex,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None  # will get filled in findHands()

    def findHands(self, img, draw=True):
        # mirror the webcam so it's natural to pose
        img = cv2.flip(img, 1)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img,
                        hand_lms,
                        self.mpHands.HAND_CONNECTIONS,
                        self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mpDraw.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
        return img

    def findLandmarkList(self):
        """
        Returns: list of [x,y,z] for the FIRST detected hand (21 points),
        or [] if no hand found.
        """
        lm_list = []
        if self.results and self.results.multi_hand_landmarks:
            first_hand = self.results.multi_hand_landmarks[0]
            for lm in first_hand.landmark:
                lm_list.append([lm.x, lm.y, lm.z])
        return lm_list


# =========================
# CSV setup
# =========================
csv_file = "gesture_data_new.csv"

# if the file doesn't exist yet, make it with header:
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ["label"]
        # x0,y0,z0 ... x20,y20,z20
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(header)
    print(f"[INIT] Created {csv_file} with header")


# =========================
# Capture loop
# =========================
cap = cv2.VideoCapture(0)
detector = HandDetector()

print("Controls:")
print("  r = rock")
print("  p = paper")
print("  s = scissor")
print("  g = game")
print("  q = quit")
print("--------------------------------------")

while True:
    ok, frame = cap.read()
    if not ok:
        print("[WARN] Camera frame not grabbed.")
        break

    # detect / draw hand
    frame = detector.findHands(frame, draw=True)

    # grab 21 landmarks from first detected hand
    lm_list = detector.findLandmarkList()

    # show instructions overlay on screen
    cv2.putText(frame, "r=rock  p=paper  s=scissor  g=game  q=quit",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2)

    # waitKey ALSO reads your key presses
    key = cv2.waitKey(1) & 0xFF
    label = None

    if key == ord('r'):
        label = "rock"
    elif key == ord('p'):
        label = "paper"
    elif key == ord('s'):
        label = "scissor"   # IMPORTANT: singular "scissor"
    elif key == ord('g'):
        label = "game"
    elif key == ord('q'):
        print("[INFO] Quit requested.")
        break

    # if we pressed something valid AND we actually have a hand
    if label and lm_list:
        row = [label]
        for (x, y, z) in lm_list:
            row.extend([x, y, z])

        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print(f"[SAVED] {label} ({len(lm_list)} landmarks)")

        # visual feedback so you know it took that sample
        cv2.putText(frame, f"SAVED: {label}",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

    # show the camera window
    cv2.imshow("Hand Gesture Collection", frame)

# cleanup
cap.release()
cv2.destroyAllWindows()

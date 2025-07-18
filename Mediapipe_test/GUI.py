import time
import cv2
import numpy as np



frame = np.zeros((500, 800, 3), dtype=np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_color = (0, 255, 0)
thickness = 2
y_offset = 100
x = 30
y_start = 100
line_height = 40

def instructions():

    Instruction_text = ("1) Make either rock, paper, and scissors gesture into the camera.",
                        "2) After two seconds raise you next gesture.",
                        "3) after three rounds the game will stop and gesture game to play again.")

    for i in range(len((Instruction_text))):
        frame_copy = frame.copy()
        for j in range(i + 1):
            y = y_start + j * line_height
            cv2.putText(frame_copy, Instruction_text[j], (x, y), font, font_scale, font_color, thickness)
        cv2.imshow("Instructions", frame_copy)
        cv2.waitKey(1)  # Refresh the OpenCV window
        time.sleep(2)  # Wait 2 seconds before showing the next line

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.waitKey(0)
cv2.destroyAllWindows()


instructions()

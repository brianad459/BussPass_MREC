import cv2
import numpy as np

# ✅ Load sprite once at the top
sprite = cv2.imread(r"C:\Users\fau_bdeloatch\PycharmProjects\PythonProject\images\gameFist-removebg-preview.png", cv2.IMREAD_UNCHANGED)
print("Sprite loaded:", sprite is not None)
print("Sprite shape:", sprite.shape)



if sprite is not None:
    sprite_h, sprite_w = sprite.shape[:2]

def overlay_sprite(frame, x, y, scale=1.0):
    global sprite, sprite_h, sprite_w

    if sprite is None:
        return frame

    # ✅ Resize the already-loaded sprite
    resized_sprite = cv2.resize(sprite, (int(sprite_w * scale), int(sprite_h * scale)))
    h, w = resized_sprite.shape[:2]

    frame_h, frame_w = frame.shape[:2]


    if y + h > frame_h or x + w > frame_w:
        return frame

    if resized_sprite.shape[2] == 4:
        overlay = resized_sprite[:, :, :3]
        alpha = resized_sprite[:, :, 3] / 255.0
        roi = frame[y:y + h, x:x + w]

        for c in range(3):
            roi[:, :, c] = (alpha * overlay[:, :, c] + (1 - alpha) * roi[:, :, c])

        frame[y:y + h, x:x + w] = roi
    else:
        frame[y:y + h, x:x + w] = resized_sprite[:, :, :3]

    return frame

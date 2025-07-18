import cv2
import numpy as np
import math

# ✅ Load sprite once at the top
sprite = cv2.imread(r"C:\Users\fau_bdeloatch\PycharmProjects\PythonProject\images\gameFist-removebg-preview.png", cv2.IMREAD_UNCHANGED)
print("Sprite loaded:", sprite is not None)
if sprite is not None:
    print("Sprite shape:", sprite.shape)
    sprite_h, sprite_w = sprite.shape[:2]

def overlay_sprite(frame, x, y, scale=1.0):
    global sprite, sprite_h, sprite_w

    if sprite is None:
        return frame

    # ✅ Resize the already-loaded sprite
    resized_sprite = cv2.resize(sprite, (int(sprite_w * scale), int(sprite_h * scale)))
    h, w = resized_sprite.shape[:2]

    frame_h, frame_w = frame.shape[:2]

    # Ensure sprite fits in frame
    if y + h > frame_h or x + w > frame_w or x < 0 or y < 0:
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

# --- Pulsing Animation Loop ---
width, height = 640, 480
center_x = width // 2
center_y = height // 2

base_scale = 1.0
pulse_amplitude = 0.5
pulse_speed = 0.05

frame_count = 0

while True:
    # Create a black background
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate pulsing scale
    scale = base_scale + pulse_amplitude * math.sin(frame_count * pulse_speed)

    # Calculate top-left position to keep sprite centered
    if sprite is not None:
        scaled_w = int(sprite_w * scale)
        scaled_h = int(sprite_h * scale)
        x = center_x - scaled_w // 2
        y = center_y - scaled_h // 2

        frame = overlay_sprite(frame, x, y, scale)

    cv2.imshow('Pulsing Sprite', frame)
    frame_count += 1
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()



import cv2
import numpy as np
import math

# ---- Sprite utilities ----
def load_sprite(path):
    sprite = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if sprite is None:
        return None, 0, 0
    h, w = sprite.shape[:2]
    return sprite, h, w

def overlay_sprite(frame, sprite, sprite_h, sprite_w, x, y, scale=1.0):
    if sprite is None:
        return frame
    resized = cv2.resize(sprite, (int(sprite_w * scale), int(sprite_h * scale)))
    h, w = resized.shape[:2]
    fh, fw = frame.shape[:2]
    if x < 0 or y < 0 or x + w > fw or y + h > fh:
        return frame

    if resized.shape[2] == 4:
        overlay = resized[:, :, :3]
        alpha = resized[:, :, 3] / 255.0
        roi = frame[y:y+h, x:x+w]
        for c in range(3):
            roi[:, :, c] = alpha * overlay[:, :, c] + (1 - alpha) * roi[:, :, c]
        frame[y:y+h, x:x+w] = roi
    else:
        frame[y:y+h, x:x+w] = resized[:, :, :3]
    return frame

def get_pulse_scale(frame_count, base_scale=1.0, pulse_amplitude=0.5, pulse_speed=0.05):
    return base_scale + pulse_amplitude * math.sin(frame_count * pulse_speed)

# ---- Optional demo (runs ONLY if you execute this file directly) ----
if __name__ == "__main__":
    path = r"C:\Users\fau_bdeloatch\PycharmProjects\PythonProject1\images\gameFist-removebg-preview.png"
    sprite, sprite_h, sprite_w = load_sprite(path)
    width, height = 640, 480
    cx, cy = width // 2, height // 2
    t = 0
    while True:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        scale = get_pulse_scale(t)
        if sprite is not None:
            sw, sh = int(sprite_w * scale), int(sprite_h * scale)
            x = cx - sw // 2
            y = cy - sh // 2
            frame = overlay_sprite(frame, sprite, sprite_h, sprite_w, x, y, scale)
        cv2.imshow("Pulsing Sprite (demo)", frame)
        t += 1
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

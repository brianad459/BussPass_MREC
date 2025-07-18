 # main.py

from ultralytics import YOLO
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import random
from trafficAPI import get_route_info, get_traffic_incidents, get_traffic_info_combined
import testing
import Speaker
from image_animation import *
import math
# Load YOLOv11 model
model = YOLO("Mediapipe_test/yolo11n.pt")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="../gesture_model_new.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Gesture labels (must match training)
import joblib  # Add to the top of the file with other imports

# Load LabelEncoder to get consistent label order
encoder = joblib.load("gesture_label_encoder.pkl")
gesture_labels = list(encoder.classes_)  # Dynamically load labels


# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Webcam
cap = cv2.VideoCapture(0)
seen_ids = set()
count = 0

# Game state variables
countdown_index = 0
countdown_start_time = None
show_countdown = False
startGame = ["ROCK", "PAPER", "SCISSORS", "ON", "GO"]
GameList = ["paper", "rock", "scissor"]
ready_for_round = False
user_move = None

user_wins = 0
computer_wins = 0
rounds = 0
counting_gestures = 0
game_started = False
who_won_called = False
instructions_spoken = False
round_start_time = None
Player = None
game_invited = False
invitation_time = 0
message_display_start = None
show_game_restart_message = False
last_detected_gesture = ""
last_confidence = 0
restart_message_start_time = None
message_duration = 10  # seconds
frame_count = 0




# Load sprite once at the top (if not already loaded)
sprite = cv2.imread(r"C:\Users\fau_bdeloatch\PycharmProjects\PythonProject\images\gameFist-removebg-preview.png", cv2.IMREAD_UNCHANGED)
sprite_h, sprite_w = sprite.shape[:2]

def overlay_sprite(frame, x, y, scale=1.0):
    global sprite, sprite_h, sprite_w
    if sprite is None:
        return frame
    resized_sprite = cv2.resize(sprite, (int(sprite_w * scale), int(sprite_h * scale)))
    h, w = resized_sprite.shape[:2]
    frame_h, frame_w = frame.shape[:2]
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

def get_pulse_scale(frame_count, base_scale=1.0, pulse_amplitude=0.5, pulse_speed=0.05):
    return base_scale + pulse_amplitude * math.sin(frame_count * pulse_speed)




# I want to say if the person at that ID is playing the game the have player on their head
def currentPLayer(person_id, game_started, annotated_frame, target_id):
    if person_id and not None and game_started:
        if person_id == target_id:
            cv2.putText(annotated_frame, "PLAYER", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (155, 245, 0), 2)


def draw_multiline_text_right_aligned(frame, text_lines, margin=20, y_start=30, line_height=20, color=(0, 255, 255), font_scale=0.5, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    frame_width = frame.shape[1]
    for i, line in enumerate(text_lines):
        text = str(line)
        (text_width, _) = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x = frame_width - text_width - margin
        y = y_start + i * line_height
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
def timer(annotated_frame, height):
    for counter in range(5, 0, -1):
        display_frame = annotated_frame.copy()
        frame_with_timer = display_frame.copy()
        cv2.putText(
            frame_with_timer,
            f"Raise your next gesture in... {counter}",
            (10, height - 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        cv2.imshow("YOLO + Gesture Recognition", frame_with_timer)
        cv2.waitKey(1)
        time.sleep(1)
def gameInvitation(person_id, gesture_name, confidence):
    global invitation_time, game_started, game_invited
    if not game_invited:
        if person_id in seen_ids:  # Using YOLO detection
            Speaker.greetingMessage("Hey, does anybody want to play a game?")
            game_invited = True
            invitation_time = time.time()

    elif not game_started:
        # Wait for hand gesture to say 'game'
        if gesture_name == "game" and confidence > 0.8:
            Speaker.startGame("Let's start the game!")
            game_started = True
            # Reset any round variables, etc.
        else:
            # Optional: Repeat the invitation every 10 seconds
            if time.time() - invitation_time > 10:
                Speaker.HowToPlay("Just make the 'game' gesture in ASL if you want to play!")
                invitation_time = time.time()


def instructions(frame):
    global instructions_spoken

    # === 1. Draw Instruction Text ===
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.7
    font_color = (212, 187, 61)
    thickness = 2
    x = 30
    y_start = 250
    line_height = 40

    Instruction_text = [
        "1) Make either rock, paper, and scissors gesture.",
        "2) After two seconds raise your next gesture.",
        "3) After three rounds, gesture 'game' to play again."
    ]

    for i, line in enumerate(Instruction_text):
        y = y_start + i * line_height
        cv2.putText(frame, line, (x, y), font, font_scale, font_color, thickness)

    # === 2. Overlay Static Sprite ===

    # === 3. Speak Instructions Once ===
    if not instructions_spoken:
        Speaker.HowToPlay(
            "1. Start by gesturing 'game' in ASL. "
            "2. Then throw your next gesture — rock, paper, or scissors. "
            "3. After three rounds, gesture 'game' again to restart."
        )
        instructions_spoken = True


def restartGameNoResponse (annotated_frame, user_move):
    last_valid_time = time.time()
    current_time = time.time()
    if user_move not in ["rock, paper, scissors"] and current_time - last_valid_time > 20:
        countdown(startGame, frame)



def whoWon(user_wins, computer_wins, frame):
    result_frame = frame.copy()
    height = frame.shape[0]
    if user_wins > computer_wins:
        msg = "User Won the Game!"
        Speaker.computerLost1("You can have that one")
        color = (0, 255, 0)
    elif computer_wins > user_wins:
        msg = "Computer Won the Game!"
        Speaker.UserLost1("Haha You lost to AI")
        color = (0, 0, 255)
    else:
        msg = "It's a Tie, that's Game!"
        Speaker.tie("ggs")
        color = (255, 255, 0)
    cv2.putText(result_frame, msg, (20, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    end_time = time.time() + 2
    while time.time() < end_time:
        cv2.imshow("YOLO + Gesture Recognition", result_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

def playingGestures(frame, GameList, user_move):
    global startGame  # e.g., ["ROCK", "PAPER", "SCISSORS", "ON", "GO"]

    height = frame.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    duration = 5
    switch_interval = 0.3
    start_time = time.time()
    last_switch_time = start_time
    final_choice = random.choice(GameList)

    # Call your countdown before starting
    timer(frame, height)

    while time.time() - start_time < duration:
        current_time = time.time()
        if current_time - last_switch_time >= switch_interval:
            final_choice = random.choice(GameList)
            last_switch_time = current_time

        display_frame = frame.copy()

        # Display user’s move
        cv2.putText(display_frame, f"Your Move: {user_move}", (20, 100),
                    font, 1.5, (0, 255, 0), 3)

        # Prompt user
        cv2.putText(display_frame, "Throw up your next gesture!",
                    (10, height - 150), font, 0.9, (0, 255, 255), 2)

        # Show computer move changing
        cv2.putText(display_frame, f"Computer picked: {final_choice}",
                    (10, height - 100), font, font_scale, (0, 0, 255), thickness)

        cv2.imshow("YOLO + Gesture Recognition", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    display_frame = frame.copy()
    cv2.putText(display_frame, f"Your Move: {user_move}", (20, 100),
                font, 1.5, (0, 255, 0), 3)
    cv2.putText(display_frame, f"Computer picked: {final_choice}",
                (10, height - 100), font, font_scale, (0, 0, 255), thickness)
    end_time = time.time() + 2
    while time.time() < end_time:
        cv2.imshow("YOLO + Gesture Recognition", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    return final_choice.lower()



def playingGameLogic(user_movement, computer_movement, annotated_frame):
    global user_wins, computer_wins, rounds
    display_frame2 = annotated_frame.copy()
    user_movement = user_movement.strip().lower()
    computer_movement = computer_movement.strip().lower()
    height, _, _ = annotated_frame.shape

    if user_movement == computer_movement:
        result_text = "It's a tie!"
    elif (
        (user_movement == "rock" and computer_movement == "scissors") or
        (user_movement == "scissors" and computer_movement == "paper") or
        (user_movement == "paper" and computer_movement == "rock")
    ):
        result_text = "User Wins!"
        user_wins += 1
    else:
        result_text = "Computer Wins!"
        computer_wins += 1

    y_position = height - 30

    cv2.putText(display_frame2, result_text, (20, height),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(display_frame2, f"User points: {user_wins}", (20, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(display_frame2, f"Computer points: {computer_wins}", (20, height - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    rounds += 1

    end_time = time.time() + 2
    while time.time() < end_time:
        cv2.imshow("YOLO + Gesture Recognition", display_frame2)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
def test_window():
    start = time.time()
    while time.time() - start < 5:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "TESTING DISPLAY", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("YOLO + Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

def countdown(startGame, frame):
    global countdown_index, countdown_start_time, show_countdown, ready_for_round, round_start_time
    if countdown_start_time is None:
        countdown_start_time = time.time()
    current_time = time.time()
    elapsed = current_time - countdown_start_time
    if countdown_index < len(startGame):
        if elapsed > countdown_index:
            word = startGame[countdown_index]
            height = frame.shape[0]
            display = frame.copy()
            cv2.putText(display, word, (100, int(height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 40, 36), 4)
            cv2.imshow("YOLO + Gesture Recognition", display)
            countdown_index += 1
    else:
        show_countdown = False
        countdown_index = 0
        countdown_start_time = None
        ready_for_round = True
        round_start_time = time.time()
def playAgain(user_movement, computer_movement, annotated_frame, rounds):
    global user_wins, computer_wins
    if rounds > 4:
        user_wins = 0
        computer_wins = 0

# === MAIN LOOP ===
while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    results = model.track(source=frame, persist=True)[0]
    person_boxes = results.boxes[results.boxes.cls == 0]
    results.boxes = results.boxes[:0]
    annotated_frame = frame.copy()
    results.plot(annotated_frame)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)


    if rounds == 3 and not who_won_called:
        whoWon(user_wins, computer_wins, annotated_frame)
        who_won_called = True

    if not game_started and not instructions_spoken:
        cv2.putText(annotated_frame, "Let's Play Rock, Paper, Scissors!", (20, 160), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 128, 255), 2)
        cv2.putText(annotated_frame, "Gesture 'Game' in ASL to start playing!", (20, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 128, 255), 2)
        instructions(annotated_frame)
        # --- Pulsing Sprite Animation under Instructions ---
        if sprite is not None:
            # Calculate pulsing scale
            scale = get_pulse_scale(frame_count)
            # Position: x=30 (left), y=after instructions (e.g., 370)
            x_sprite = 30
            y_sprite = 370
            annotated_frame = overlay_sprite(annotated_frame, x_sprite, y_sprite, scale)
        frame_count += 1

    #text_info = get_traffic_info_combined()
    #draw_multiline_text_right_aligned(annotated_frame, text_info)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmark_list = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
            input_data = np.array([landmark_list], dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])
            predicted_label = np.argmax(predictions)
            gesture_name = gesture_labels[predicted_label]
            confidence = predictions[0][predicted_label]
            last_confidence =  confidence
            last_detected_gesture = gesture_name  # Track it globally

            if rounds >= 3:
                if gesture_name == "game" and confidence > 0.85:
                    game_started = True
                    countdown_start_time = None
                    show_countdown = True
                    user_wins = 0
                    computer_wins = 0
                    rounds = 0
                    who_won_called = False
                    testing.get_traffic_incidents()
                    show_game_restart_message = False
                    message_display_start = None
                else:
                    if not show_game_restart_message:
                       show_game_restart_message = True
                       message_display_start = time.time()

                continue

            if show_countdown:
                countdown(startGame, annotated_frame)
                continue

            elif gesture_name == "game" and confidence > 0.85 and not game_started:
                game_started = True
                countdown_start_time = None
                show_countdown = True
                continue

            elif game_started and not show_countdown and not ready_for_round:
                if confidence > 0.85 and gesture_name in ["paper", "rock", "scissor"]:
                    user_move = gesture_name.capitalize()
                    ready_for_round = True
                    continue

            elif ready_for_round:
                if confidence > 0.85 and gesture_name in ["paper", "rock", "scissor"]:
                    user_move = gesture_name.capitalize()
                    last_move_time = time.time()
                    #test_window()
                    computer_move = playingGestures(annotated_frame.copy(), GameList, user_move).capitalize()
                    playingGameLogic(user_move, computer_move, annotated_frame)
                    restartGameNoResponse(annotated_frame, user_move)
                    ready_for_round = False
                    user_move = None
                    counting_gestures += 1

    for box in person_boxes:
        if box.id is None:
            continue
        person_id = int(box.id[0].item())
        currentPLayer(person_id, game_started, annotated_frame, person_id)
        if person_id not in seen_ids:
            seen_ids.add(person_id)
            count += 1
        gameInvitation(
            person_id,
            locals().get('gesture_name', ''),
            locals().get('confidence', 0.0)
        )
    results.boxes = results.boxes[:0] # This doesn't include any bounding boxes on anything
    annotated_frame = results.plot()


    if game_started:
        if rounds < 3:
            cv2.putText(annotated_frame, f"Round: {rounds + 1}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif game_started and rounds == 3:
            cv2.putText(annotated_frame, "All rounds complete!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    start_message_start_time = time.time()
    mess_dur = 0



    # In your main loop
    if show_game_restart_message and last_detected_gesture != "game":
        if restart_message_start_time is None:
            restart_message_start_time = time.time()

        elapsed = time.time() - restart_message_start_time
        if elapsed < message_duration:
            # Display message
            cv2.putText(annotated_frame, "All rounds played.", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, "Gesture 'GAME' to play again!", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

            Instruction_text1 = "1) Make either rock, paper, and scissors gesture."
            Instruction_text2 = "2) After three seconds raise your next gesture."
            Instruction_text3 = "3) After three seconds play again."
            cv2.putText(annotated_frame, Instruction_text1, (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, Instruction_text2, (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, Instruction_text3, (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # 10 seconds passed, stop showing message
            show_game_restart_message = False
            restart_message_start_time = None
    else:
        restart_message_start_time = None  # Also reset if we’re not in the condition

    # Reset if "game" gesture is shown
    if last_detected_gesture.lower() == "game" and last_confidence > 0.90:
        show_game_restart_message = False
        restart_message_start_time = None

    gameInvitation( locals().get('person_id', 0), locals().get('gesture_name', ''), locals().get('confidence', 0.0))



    cv2.putText(annotated_frame, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("YOLO + Gesture Recognition", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
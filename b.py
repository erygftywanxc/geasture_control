# pip install opencv-python mediapipe pycaw numpy keyboard screen-brightness-control
import cv2
import numpy as np
import mediapipe as mp
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
import keyboard
import time
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    model_complexity=1
)
mp_drawing = mp.solutions.drawing_utils

# Initialize audio control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, 0x17, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

class GestureControl:
    def __init__(self):
        self.vol_percent = volume.GetMasterVolumeLevelScalar() * 100
        self.prev_hand_pos = None
        self.history = []
        self.last_gesture_time = 0
        self.current_gesture = "None"
        self.brightness_percent = sbc.get_brightness()[0]
        self.min_brightness_dist = 0.05  # Adjust based on your hand size
        self.max_brightness_dist = 0.25   # Adjust based on your hand size

    def update_volume(self, delta_y):
        self.vol_percent = np.clip(self.vol_percent + delta_y * 150, 0, 100)
        volume.SetMasterVolumeLevelScalar(self.vol_percent / 100, None)

    def update_brightness(self, distance):
        self.brightness_percent = np.interp(
            distance,
            [self.min_brightness_dist, self.max_brightness_dist],
            [0, 100]
        )
        self.brightness_percent = np.clip(self.brightness_percent, 0, 100)
        try:
            sbc.set_brightness(int(self.brightness_percent))
        except Exception as e:
            print(f"Brightness error: {e}")

def get_finger_state(landmarks, finger_tip, finger_pip):
    return landmarks[finger_tip].y < landmarks[finger_pip].y

def draw_gesture_help(image):
    help_text = [
        "Gesture Guide:",
        "Volume: 4 fingers up + vertical movement",
        "App Switch: 2 fingers up + horizontal swipe",
        "Brightness: Thumb+Index spread + distance",
        "Calibration: Thumb+Index circle (hold 3s)"
    ]
    y_start = 30
    for i, text in enumerate(help_text):
        color = (0, 255, 0) if i == 0 else (200, 200, 200)
        cv2.putText(image, text, (10, y_start + i*30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0)
    controller = GestureControl()
    calibration_counter = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark

            # Get finger states
            fingers = {
                'thumb': get_finger_state(landmarks, 4, 2),
                'index': get_finger_state(landmarks, 8, 6),
                'middle': get_finger_state(landmarks, 12, 10),
                'ring': get_finger_state(landmarks, 16, 14),
                'pinky': get_finger_state(landmarks, 20, 18)
            }

            current_pos = np.mean([(lm.x, lm.y) for lm in [
                landmarks[4], landmarks[8], 
                landmarks[12], landmarks[16], landmarks[20]
            ]], axis=0)

            # Volume Control (4 fingers)
            if sum([fingers['index'], fingers['middle'], 
                   fingers['ring'], fingers['pinky']]) == 4:
                controller.current_gesture = "Volume Control"
                if controller.prev_hand_pos is not None:
                    delta_y = controller.prev_hand_pos[1] - current_pos[1]
                    controller.update_volume(delta_y)
                
                # Volume bar
                cv2.rectangle(image, (20, 20), (50, 400), (0, 255, 0), 2)
                vol_height = int(380 * (controller.vol_percent/100))
                cv2.rectangle(image, (20, 400 - vol_height), 
                            (50, 400), (0, 255, 0), cv2.FILLED)

            # Brightness Control (Thumb + Index)
            elif fingers['thumb'] and fingers['index'] and not \
                 any([fingers['middle'], fingers['ring'], fingers['pinky']]):
                controller.current_gesture = "Brightness Control"
                thumb = landmarks[4]
                index = landmarks[8]
                distance = math.hypot(thumb.x-index.x, thumb.y-index.y)
                controller.update_brightness(distance)
                
                # Draw connection line
                cv2.line(image,
                    (int(thumb.x * image.shape[1]), int(thumb.y * image.shape[0])),
                    (int(index.x * image.shape[1]), int(index.y * image.shape[0])),
                    (255, 0, 0), 2)
                # Brightness percentage
                cv2.putText(image, f'{int(controller.brightness_percent)}%', 
                          (int(thumb.x * image.shape[1])+10, int(thumb.y * image.shape[0])),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # App Switching (2 fingers)
            elif sum([fingers['index'], fingers['middle']]) == 2 and not \
                 any([fingers['ring'], fingers['pinky']]):
                controller.current_gesture = "App Switch Ready"
                if controller.prev_hand_pos is not None:
                    delta_x = current_pos[0] - controller.prev_hand_pos[0]
                    if time.time() - controller.last_gesture_time > 0.5:
                        if delta_x > 0.1:
                            keyboard.send('alt+tab')
                            controller.last_gesture_time = time.time()
                        elif delta_x < -0.1:
                            keyboard.send('alt+shift+tab')
                            controller.last_gesture_time = time.time()

            # Store hand position history
            controller.history.append(current_pos)
            if len(controller.history) > 5:
                controller.history.pop(0)
            controller.prev_hand_pos = np.mean(controller.history, axis=0)

            # Draw landmarks
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0) if "Volume" in controller.current_gesture else 
                                      (255,0,0) if "Brightness" in controller.current_gesture else 
                                      (0,0,255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2))

        # Draw UI
        draw_gesture_help(image)
        cv2.putText(image, f"Active Gesture: {controller.current_gesture}", 
                   (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Advanced Gesture Control', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
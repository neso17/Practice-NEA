import mediapipe as mp
import cv2
import math
import numpy as np
import pygame

class Gesture:
    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h

        self.running = True
        self.cap = cv2.VideoCapture(0)

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.4
        )

        self.mp_draw = mp.solutions.drawing_utils

    def processing(self):
        if self.running:
            success, raw = self.cap.read()
            if not success:
                return None
            mirrored = cv2.flip(raw, 1)
            display = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB) #rgb for pygame display, bgr for mediapipe
            colored = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(colored, (self.screen_w, self.screen_h))

            landmarks = self.hands.process(resized)
            display = resized.copy()

            return landmarks, display

    def drawHand(self, hand_landmarks, frame):
        black_bg = np.zeros_like(frame)
        # draw points + connections on black bg
        self.mp_draw.draw_landmarks(
            black_bg, 
            hand_landmarks, 
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=5),
            self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
        )

        display_frame = cv2.transpose(black_bg) #pygame reads horizontal , cv2 reads vertical 
        pygame_surface = pygame.surfarray.make_surface(display_frame)
        pygame_surface = pygame.transform.scale(pygame_surface, (self.screen_w, self.screen_h))
        pygame_surface.set_colorkey((0, 0, 0))  # Set black as transparent
        return pygame_surface

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


    def detect_gesture(self, hand_landmarks, w, h):
        """Analyzes landmarks and returns: 'CLICK', 'FIST', or 'POINT'"""
        thumb = hand_landmarks.landmark[4]
        index = hand_landmarks.landmark[8]
        middle = hand_landmarks.landmark[12]
        ring = hand_landmarks.landmark[16]
        pinky = hand_landmarks.landmark[20]

        thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
        index_x, index_y = int(index.x * w), int(index.y * h)

        # 1) CLICK- Thumb and Index touching
        distance = math.hypot(index_x - thumb_x, index_y - thumb_y)
        if distance < 30:  # CLICK_THRESHOLD_PIXELS
            return "CLICK"

        # 2) FIST- Fingers below knuckles
        if (index.y > hand_landmarks.landmark[6].y and 
            middle.y > hand_landmarks.landmark[10].y and 
            ring.y > hand_landmarks.landmark[14].y and 
            pinky.y > hand_landmarks.landmark[18].y):
            return "FIST"

        # 3) POINT- Default state 
        return "POINT"


if __name__ == "__main__":
    # 1. Quick Pygame initialization just for testing the display window
    pygame.init()
    screen = pygame.display.set_mode((1400, 1000))
    clock = pygame.time.Clock()

    gesture = Gesture(1400, 1000)
    print("Window running. Press the 'X' button or close the window to stop.")

    running = True
    while running:
        clock.tick(30)  # Standard camera frame rate limitation

        # Catch window close commands
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((40, 60, 160))

        results, display = gesture.processing()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                skeleton_overlay = gesture.drawHand(hand_landmarks, display) #make surface
                screen.blit(skeleton_overlay, (0, 0))

                action = gesture.detect_gesture(hand_landmarks, 1400, 1000)
                print(f"Tracking Hand - Action detected: {action}")

        pygame.display.flip() #pygame updates display

    gesture.release()
    pygame.quit()
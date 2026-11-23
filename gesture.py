import threading

import mediapipe as mp
import cv2
import math
import numpy as np
import pygame
import time
import queue

class Gesture:
    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h

        self.running = True
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.screen_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.screen_h) # resizing once here, rather than every time in update() for performance

        # Initialise MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.4
        )

        #self.mp_draw = mp.solutions.drawing_utils
        self.joints = []
        self.skeleton_surface = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)

        self.indexF_y = 0
        self.currentG = "NONE"
        self.hand_detected = False

        self.indexF_log = queue.Queue(maxsize=5)  # for swipe/ hover checking 
        self.hover_start_time = 0
        self.hRAD = 30  # hover radius 
        self.hTHRESHOLD = 1.0  # s
        self.sSPEED = 50

        # Thread-safe shared state- 
        self.lock = threading.Lock() 
        self.last_seen_time = time.time() # starts timer for last time hand is seen
        self.thread = threading.Thread(target=self._update_thread, daemon=True)
        self.thread.start()


    def get_gesture(self):
        with self.lock: 
            return self.currentG

    def get_indexF_y(self):
        with self.lock:
            return self.indexF_y

    def get_joints(self):
        with self.lock:
            return self.joints

    def is_present(self):
        with self.lock:
            return self.hand_detected or (time.time() - self.last_seen_time) < 1

    def processing(self): #process feed and return landmarks
        if not self.running:
            return None
        success, raw = self.cap.read()
        if not success:
            return None
        
        mirrored = cv2.flip(raw, 1)
        coloured = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)

        landmarks = self.hands.process(coloured)

        return landmarks # landmarks for gesture detection

    def _update_thread(self):
        while self.running:
            results = self.processing()
            new_joints = []
            new_indexF_y = 0
            new_currentG = "NONE"
            detected = False

            if results and results.multi_hand_landmarks:
                detected = True
                hand_landmarks = results.multi_hand_landmarks[0]
    
                for lm in hand_landmarks.landmark:
                    new_joints.append((int(lm.x * self.screen_w), int(lm.y * self.screen_h)))
    
                new_indexF_y = hand_landmarks.landmark[8].y
                new_currentG = self.detect_gesture(new_joints)
                self.hand_detected = True
                
            with self.lock:
                self.joints = new_joints
                self.indexF_y = new_indexF_y
                self.currentG = new_currentG
                self.hand_detected = detected
                if detected:
                    self.last_seen_time = time.time()  # Update last seen time when hand is detected

    def draw_hand(self, display): #draws skeleton overlay
        # make surface of correct size
        self.skeleton_surface.fill((0, 0, 0, 0))  # Clear the surface with transparency
        
        #landmark is an object with attrivutes x, y, z, visibility and presence
        joints = self.get_joints()
        if not joints:
            return self.skeleton_surface

        # bone connections
        for startI, endI in self.mp_hands.HAND_CONNECTIONS:
            pygame.draw.line(self.skeleton_surface, (255, 255, 255), joints[startI], joints[endI], 2)

        # joint circles
        for joint in joints:
            pygame.draw.circle(self.skeleton_surface, (0, 255, 0), joint, 5)

        display.blit(self.skeleton_surface, (0, 0))

    def release(self):
        self.running = False
        self.thread.join()  # Wait for the thread to finish
        self.cap.release()
        cv2.destroyAllWindows()


    def detect_gesture(self, joints): #inside thread so pass in values instead of getters
        if not joints:
            return "NONE"
        thumbT, indexT, middleT, ringT, pinkyT = joints[4], joints[8], joints[12], joints[16], joints[20]
        indexM, middleM, ringM, pinkyM = joints[6], joints[10], joints[14], joints[18]


        # 1) CLICK- Thumb and Index touching
        distance = math.hypot(indexT[0] - thumbT[0], indexT[1] - thumbT[1])
        if distance < 30:
            return "CLICK"

        # 2) FIST- Fingers below knuckles
        if (indexT[1] > indexM[1] and
            middleT[1] > middleM[1] and
            ringT[1] > ringM[1] and
            pinkyT[1] > pinkyM[1]):
            return "FIST"

        # 3) POINT- Index finger extended, others curled
        if (indexT[1] < indexM[1] and
            middleT[1] > middleM[1] and 
            ringT[1] > ringM[1] and 
            pinkyT[1] > pinkyM[1]):
            return "POINT"

        # 4) Palm open- All fingers extended
        if (indexT[1] < indexM[1] and
            middleT[1] < middleM[1] and 
            ringT[1] < ringM[1] and 
            pinkyT[1] < pinkyM[1]):
            return "PALM"

        else:
            return "NONE"
    
    def detect_action(self, gest, hand_landmarks, w, h): #detctions motion

        if gest != "POINT":
            return "NONE"
        
        indexF_log = queue.Queue(maxsize=5) 
        for i in range(5):
            indexF_log.put(hand_landmarks.landmark[8])  # Index fingertip

        # 1) Swipe 

        # 2) Hover
        for i in range(5):
            if abs(indexF_log.queue[i].x - indexF_log.queue[i+1].x) < self.hRAD:
                isHovering = True
                
            else:
                isHovering = False
                self.hover_start_time = time.time()  # Reset hover timer

  

if __name__ == "__main__":
    pygame.init()
    WIDTH, HEIGHT = 1600, 1080
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Hand Tracker Test")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 28, bold=True)

    tracker = Gesture(WIDTH, HEIGHT)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((20, 20, 20))

        tracker.draw_hand(screen)

        joints = tracker.get_joints()
        pointer_pos = joints[8] if joints else "N/A"
        hud_text = font.render(
            f"Gesture: {tracker.get_gesture()}  |  Pointer: {pointer_pos}",
            True,
            (255, 255, 0),
        )
        screen.blit(hud_text, (20, 20))

        pygame.display.flip()
        clock.tick(60)

    tracker.release()
    pygame.quit()
    # 1. Quick Pygame initialization just for testing the display window
    pygame.init()
    WIDTH, HEIGHT = 1600, 1080
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Hand Tracker Test")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 28, bold=True)

    tracker = Gesture(WIDTH, HEIGHT)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((20, 20, 20))

        # 1. Update tracker (draws camera + hand skeleton onto screen)
        tracker.draw_hand(screen)

        # 2. Draw debug HUD on top to verify gesture detection & pointer coords
        pointer_pos = tracker.joints[8] if tracker.joints else "N/A"
        hud_text = font.render(
            f"Gesture: {tracker.get_gesture()}  |  Pointer: {pointer_pos}",
            True,
            (255, 255, 0),
        )
        screen.blit(hud_text, (20, 20))

        pygame.display.flip()
        clock.tick(60)

    tracker.release()
    pygame.quit()

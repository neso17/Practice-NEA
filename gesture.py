import mediapipe as mp
import cv2

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

    def preparing(self):
        if self.running:
            success, raw = self.cap.read()
            if not success:
                return None
            mirrored = cv2.flip(raw, 1)
            colored = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(colored, (self.screen_w, self.screen_h))
            return resized
        return None

    def processing(self, frame):
        results = self.hands.process(frame)
        display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(display_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return results, display_frame


    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    gesture = Gesture(1400, 1000)
    
    while True:
        resized_frame = gesture.preparing()
        if resized_frame is None:
            print("Failed to capture frame from camera.")
            break
            
        results, frame = gesture.processing(resized_frame)
        if frame is None:
            print("Failed to process frame.")
            break
            
        cv2.imshow("Gesture Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    gesture.release()
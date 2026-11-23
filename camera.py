import cv2
import threading
from gesture import Gesture

class Camera:
    def __init__(self, screen_w=1600, screen_h=1000):
        self.scr_w = screen_w
        self.scr_h = screen_h
        self.cap = cv2.VideoCapture(0)
        self.running = True

        self.action = "NONE"

        self.gesture = Gesture(self.scr_w, self.scr_h)


    def get_tracking_data(self):
        """Returns simple values directly to game loops."""
        return self.indexF_y, self.curr_gesture, self.joint_points

    def release(self): 
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        if self.cap.isOpened():
            self.cap.release()
        self.gesture.close()
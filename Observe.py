import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import imutils as util
import cv2 as cv

hand_cascade = cv.CascadeClassifier("miscellaneous/data/haarcascades/haarcascade_hand.xml")
face_cascade = cv.CascadeClassifier("miscellaneous/data/haarcascades/haarcascade_frontalface_default.xml")
smile_cascade = cv.CascadeClassifier("miscellaneous/data/haarcascades/haarcascade_smile.xml")
cascade_eye = cv.CascadeClassifier("miscellaneous/data/haarcascades/haarcascade_eye.xml")


class Vision:
    def __init__(self):
        self.cam = cv.VideoCapture(0)
        if not self.cam.isOpened():
            raise RuntimeError("Error opening your camera")

    def detect_face(self, frame):
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)
        faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 130, 0), 2)
        self.display("Face", frame)

    def detect_smile(self, frame):
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)
        faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_gray = frame_gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
            for (sx, sy, sw, sh) in smiles:
                cv.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
        self.display("Face", frame)

    def detect_eyes(self, frame):
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)
        faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_gray = frame_gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = cascade_eye.detectMultiScale(roi_gray, 1.2, 18)
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 180, 60), 2)
        self.display("Face", frame)

    def detect_hands(self, frame):
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)
        hands = hand_cascade.detectMultiScale(frame_gray, 1.5, 5)
        for (x, y, w, h) in hands:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 130, 0), 2)
        self.display("Face", frame)

    @staticmethod
    def display(window, source):
        cv.namedWindow(window)
        cv.imshow(window, source)

    def start(self):
        print("start")
        while True:
            status, img = self.cam.read()
            if not status:
                break
            #self.detect_face(img)
            #self.detect_smile(img)
            #self.detect_eyes(img)
            self.detect_hands(img)
            if cv.waitKey(1) & 0xFF == ord('q'):
                self.cam.release()
                cv.destroyAllWindows()
                break


if __name__ == "__main__":
    test = Vision()
    test.start()

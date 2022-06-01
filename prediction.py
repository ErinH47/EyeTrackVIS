from matplotlib.font_manager import _Weight
import numpy as np
import pyautogui
import os
import cv2
from tensorflow import keras

cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
video_capture = cv2.VideoCapture(0)

def normalize(x):
  minn, maxx = x.min(), x.max()
  return (x - minn) / (maxx - minn)
  
def scan(image_size=(32, 32)):
  _, frame = video_capture.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  boxes = cascade.detectMultiScale(gray, 1.3, 10)
  if len(boxes) == 2:
    eyes = []
    for box in boxes:
      x, y, w, h = box
      eye = frame[y:y + h, x:x + w]
      eye = cv2.resize(eye, image_size)
      eye = normalize(eye)
      eye = eye[10:-10, 5:-5]
      eyes.append(eye)
    return (np.hstack(eyes) * 255).astype(np.uint8)
  else:
    return None

width, height = 2559, 1439
model = keras.models.load_model("eye_track_model")


# init kalman filter object
kalman_noise = 0.03

kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)

kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * kalman_noise

prediction = np.zeros((2, 1), np.float32)

# initialize drawing
img = np.zeros((width,height),np.uint8)
before_kalman_color = (0,255,0)
before_kalman_thickness = 2
after_kalman_color = (255,0,0)
after_kalman_thickness = 3
x_old, y_old = -1, -1
prediction_old = -np.ones((2, 1), np.float32)

while True:
  eyes = scan()
  if not eyes is None:
    eyes = np.expand_dims(eyes / 255.0, axis = 0)
    x, y = model.predict(eyes)[0]
    kalman.correct(np.array([x,y]))
    prediction = kalman.predict()
    pyautogui.moveTo(prediction[0] * width, prediction[1] * height)
    
    if x_old >= 0: # draw
      cv2.line(img, (x_old, y_old), (x, y), before_kalman_color, before_kalman_thickness)
      cv2.line(img, prediction_old, prediction, after_kalman_color, after_kalman_thickness)
    x_old, y_old = x, y
    prediction_old = prediction
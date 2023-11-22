# importing the modules

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# loading the training model

detector=tf.keras.models.load_model("emotion_detector.keras")

# defining class names

class_names=['angry', 'happy', 'relaxed', 'sad']

# using model with webcam

cam=cv2.VideoCapture(0)
while True:
    _,frame=cam.read()
    frame=cv2.flip(frame,1)
    image=cv2.resize(frame,(192,192))
    prediction=class_names[np.argmax(detector.predict(image.reshape(1,192,192,3),verbose=0))]
    cv2.putText(frame, prediction, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA) 
    cv2.imshow("image",frame)
    if cv2.waitKey(1)==27:
        break
cam.release()
cv2.destroyAllWindows()
import numpy as np
import tensorflow as tf
import cv2 as cv

def infer_with_cam(model):
    font = cv.FONT_HERSHEY_COMPLEX
    vid = cv.VideoCapture(0)

    while True:
        _, frame = vid.read()
        frame_res = cv.resize(frame, (64, 64))
        frame_res = frame_res[np.newaxis, :]
        preds = model.predict(frame_res)#[0][0]

        print (preds)
        if preds[0][0] == 1:
            text = 'Smiling!!!'
        else:
            text = 'Not Smiling!'
        cv.putText(frame, text, (10, 80), font, 0.7, (255, 255, 0), 2, cv.LINE_AA)

        cv.imshow('Happy House', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv.destroyAllWindows()

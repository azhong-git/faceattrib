import cv2
import numpy as np
import time
face_detector = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
w_ext_ratio = [0.3, 0.3]
h_ext_ratio = [0.4, 0.20]
take_selfie = False

from labels import LABELS_40
import keras
from keras.models import load_model
# definte relu6
from tensorflow.python.keras import backend as K
def relu6(x):
    return K.relu(x, max_value=6)

model = load_model('models/mobilenet_1.0_224_2018_06_18_16_08_08/face_attrib_mobilenet_1.0_224.10-0.18-0.17.hdf5',
                   custom_objects={'relu6': relu6})

while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    width = bgr_image.shape[1]
    height = bgr_image.shape[0]
    if take_selfie:
        face_rects = face_detector.detectMultiScale(gray_image, 1.3, 5)
        if len(face_rects) == 0:
            cv2.imshow('window_frame', bgr_image)
            continue

        x, y, w, h = face_rects[0]
        assert w == h
        w_ext = [int(w*w_ext_ratio[i]) for i in range(2)]
        h_ext = [int(h*h_ext_ratio[i]) for i in range(2)]
        w_full = w + w_ext[0] + w_ext[1]
        h_full = h + h_ext[0] + h_ext[1]
        face_crop = np.zeros([h_full,w_full,3], dtype=int)
        x_from     = max(x-w_ext[0], 0)
        x_from_out = max(w_ext[0]-x, 0)
        y_from     = max(y-h_ext[0], 0)
        y_from_out = max(h_ext[0]-y, 0)
        x_to = min(x+w+w_ext[1], width - 1)
        y_to = min(y+h+h_ext[1], height - 1)
        face_crop[y_from_out:(y_to-y_from+y_from_out), x_from_out:(x_to-x_from+x_from_out)] = rgb_image[y_from:y_to, x_from:x_to]
        face_crop = face_crop.astype(np.uint8)
        face_crop = cv2.resize(face_crop, (224, 224))
        face_in = np.array(face_crop)
        face_in = (face_in / 255.0 - 0.5) * 2.0
        face_in = np.expand_dims(face_in, axis=0)
        start = time.time()
        val_predicted = model.predict(face_in)
        print('One frame took {:.2f} ms'.format((time.time()-start)*1e3))

        sorted_inds = np.argsort(val_predicted[0])[::-1]
        cv2.rectangle(bgr_image, (x,y), (x+w, y+h), (255, 255, 255), 2)
        face_crop_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)

        cv2.imshow('face_crop', face_crop_bgr)
        print('---------------------------------------------------------------------------------')
        for i in range(len(LABELS_40)):
            print('{}, {:.02f}'.format(LABELS_40[sorted_inds[i]], val_predicted[0][sorted_inds[i]]))
        take_selfie = False

    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        take_selfie = True

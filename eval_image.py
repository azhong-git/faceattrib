import sys
import cv2
import numpy as np

face = cv2.imread(sys.argv[1])
face = cv2.resize(face, (224, 224))
face = np.array(face)
face = (face / 255.0 - 0.5) * 2.0
faces = np.expand_dims(face, axis=0)

from labels import LABELS
import keras
from keras.models import load_model
# definte relu6
from tensorflow.python.keras import backend as K
def relu6(x):
    return K.relu(x, max_value=6)

model = load_model('models/mobilenet_1.0_224_2018_06_13_17_27_59/face_attrib_mobilenet_1.0_224.25-0.32-0.08.hdf5',
                   custom_objects={'relu6': relu6})
val_predicted = model.predict(faces)

sorted_inds = np.argsort(val_predicted[0])[::-1]
for i in range(len(LABELS)):
    print('{}, {:.02f}'.format(LABELS[sorted_inds[i]], val_predicted[0][sorted_inds[i]]))

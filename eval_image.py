import sys
import cv2
import numpy as np
import time

face = cv2.imread(sys.argv[1])
face = cv2.resize(face, (224, 224))
# face = cv2.resize(face, (192, 192))
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
face = np.array(face)
face = (face / 255.0 - 0.5) * 2.0
faces = np.expand_dims(face, axis=0)

from labels import LABELS_40
import keras
from keras.models import load_model
# definte relu6
from tensorflow.python.keras import backend as K
def relu6(x):
    return K.relu(x, max_value=6)

# model = load_model('models/mobilenet_1.0_224_2018_06_18_16_08_08/face_attrib_mobilenet_1.0_224.10-0.18-0.17.hdf5',
#                    custom_objects={'relu6': relu6})
# model = load_model('models/mobilenet_1.0_192_2018_06_19_14_57_50/face_attrib_mobilenet_1.0_192.09-0.19-0.18.hdf5',
#                    custom_objects={'relu6': relu6})
# model = load_model('models/mobilenet_0.75_224_2018_06_20_11_40_04/face_attrib_mobilenet_0.75_224.11-0.19-0.17.hdf5',
#                    custom_objects={'relu6': relu6})
model = load_model('models/mobilenet_0.5_224_2018_06_20_17_04_49/face_attrib_mobilenet_0.5_224.14-0.19-0.18.hdf5',
                   custom_objects={'relu6': relu6})
times_ms = []
for i in range(2):
    start = time.time()
    val_predicted = model.predict(faces)
    times_ms.append((time.time()-start)*1e3)

sorted_inds = np.argsort(val_predicted[0])[::-1]
for i in range(len(LABELS_40)):
    print('{}, {:.02f}'.format(LABELS_40[sorted_inds[i]], val_predicted[0][sorted_inds[i]]))
print(times_ms)

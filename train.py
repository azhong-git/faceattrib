from keras.applications.mobilenet import MobileNet
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, GlobalAveragePooling2D, Reshape, Dropout, Conv2D, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import datetime
import os
import pickle
import numpy as np

from data import load_lfw_data, split_data_by_ratio, MacOSFile

batch_size = 64
num_epochs = 100000
use_imagenet_weight = True
input_shape = (224, 224, 3)
alpha=1.0
limit = -1
validation_split = .2
verbose = 1
num_classes = 7
patience = 50
use_generator=False

# data generator
data_generator = ImageDataGenerator(featurewise_center=False,
                                    featurewise_std_normalization=False,
                                    rotation_range=10,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=.1,
                                    horizontal_flip=True)

attributes_path = '/Users/azhong/face/data/lfw/lfw_attributes.txt'
images_path = '/Users/azhong/face/data/lfw/all'
data_file = 'data/data_{}_{}_{}_limit_{}.p'.format(input_shape[0], input_shape[1], input_shape[2], limit)
if not os.path.isfile(data_file):
    faces, attributes, raw_attributes, labels = load_lfw_data(images_path, attributes_path, input_shape, limit=limit)
    attributes = np.array(attributes)
    raw_attributes = np.array(raw_attributes)
    pickle.dump({'faces': faces, 'attributes': attributes, 'raw_attributes': raw_attributes, 'labels': labels},
                MacOSFile(open(data_file, 'wb')), protocol=4)
else:
    data = pickle.load(MacOSFile(open(data_file, 'rb')))
    faces = data['faces']
    attributes = data['attributes']
    raw_attributes = data['raw_attributes']
    labels = data['labels']

now = datetime.datetime.now()
model_name = 'mobilenet_{}_{}'.format(alpha, input_shape[0])
base_path = 'models/' + model_name + '_'
base_path += now.strftime("%Y_%m_%d_%H_%M_%S") + '/'
if not os.path.exists(base_path):
    os.makedirs(base_path)

if use_imagenet_weight:
    mobilenet_model = MobileNet(input_shape=input_shape, alpha=alpha, depth_multiplier=1, dropout=1e-3, include_top=False,
                      weights='imagenet', input_tensor=None, pooling=None, classes=len(labels))
    input = Input(shape = input_shape)
    classes = len(labels)
    dropout = 1e-3
    shape = (1, 1, int(1024 * alpha))
    x = mobilenet_model(input)
    x = GlobalAveragePooling2D()(x)
    x = Reshape(shape, name='reshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(classes, (1, 1),
               padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)
    model = Model(input, x)
else:
    model = MobileNet(input_shape=input_shape, alpha=alpha, depth_multiplier=1, dropout=1e-3, include_top=True,
                      weights=None, input_tensor=None, pooling=None, classes=len(labels))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

log_file_path = base_path + 'face_attrib_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience/4), verbose=1)
trained_models_path = base_path + 'face_attrib_' + model_name
model_names = trained_models_path + '.{epoch:02d}-{val_loss:.2f}-{loss:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                   save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

train_data, val_data = split_data_by_ratio(faces, attributes)
train_images, train_vals = train_data
if use_generator:
    model.fit_generator(data_generator.flow(train_images, train_vals, batch_size),
                        steps_per_epoch = len(train_images)/batch_size,
                        epochs = num_epochs,
                        callbacks = callbacks,
                        verbose = 1,
                        validation_data = val_data
    )
else:
    model.fit(x = train_images,
              y = train_vals,
              batch_size = batch_size,
              epochs = num_epochs,
              callbacks = callbacks,
              verbose = 1,
              validation_data = val_data
    )

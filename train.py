# use only one gpu
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from keras.applications.mobilenet import MobileNet
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, GlobalAveragePooling2D, Reshape, Dropout, Conv2D, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allocator_type = 'BFC'
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# set_session(tf.Session(config=config))

import datetime
import os
import pickle
import numpy as np

from data import load_data, split_data_by_ratio
from sys import platform
if platform == 'darwin':
    from data import MacOSFile

batch_size = 64
num_epochs = 100000
input_shape = (224, 224, 3)
alpha=1.0
validation_split = .2
verbose = 1
patience = 50
use_generator=True
data_type = 'celeba'
if data_type == 'lfw':
    data_dir = '/data/lfw'
    images_path = os.path.join(data_dir, 'lfw_all_funneled_face_crop_l_0.3_r_0.3_t_0.4_d_0.2/all')
    attributes_path = os.path.join(data_dir, 'lfw_header_lines_45.p')
elif data_type == 'celeba':
    data_dir = '/data/celeba'
    images_path = os.path.join(data_dir, 'Img/img_align_celeba/')
    attributes_path = os.path.join('.', 'celeba_header_lines_40.p')
else:
    assert False, data_type + ' not supported'

local_data_path = './data'
limit=-1

if not os.path.isdir(local_data_path):
    os.makedirs(local_data_path)
data_file = os.path.join(local_data_path,
                         '{}_{}_{}_{}_limit_{}.p'.format(data_type, input_shape[0], input_shape[1], input_shape[2], limit))

if not os.path.isfile(data_file):
    faces, attributes, labels = load_data(images_path, attributes_path, input_shape, random=False, limit=limit)
    attributes = np.array(attributes)
    try:
        if platform == 'darwin':
            fo = MacOSFile(open(data_file, 'wb'))
        else:
            fo = open(data_file, 'wb')
        pickle.dump({'faces': faces, 'attributes': attributes, 'labels': labels},
                    fo, protocol=4)
    except:
        print('somehow writing to pickle failed')
else:
    if platform == 'darwin':
        fi = MacOSFile(open(data_file, 'rb'))
    else:
        fi = open(data_file, 'rb')
    data = pickle.load(fi)
    faces = data['faces']
    attributes = data['attributes']
    labels = data['labels']

# splitted_data = split_data_by_ratio(faces, attributes, [0.803409, 0.0980663281, 0.0985246719])
splitted_data = split_data_by_ratio(faces, attributes, [0.8, 0.1, 0.1])
assert False
train_images, train_vals = splitted_data[0]
val_data = splitted_data[1]
test_data = splitted_data[2]

# data generator
data_generator = ImageDataGenerator(featurewise_center=False,
                                    featurewise_std_normalization=False,
                                    rotation_range=10,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=.1,
                                    horizontal_flip=True)

now = datetime.datetime.now()
model_name = 'mobilenet_{}_{}'.format(alpha, input_shape[0])
base_path = 'models/' + model_name + '_'
base_path += now.strftime("%Y_%m_%d_%H_%M_%S") + '/'
if not os.path.exists(base_path):
    os.makedirs(base_path)


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
x = Activation('sigmoid', name='act_softmax')(x)
x = Reshape((classes,), name='reshape_2')(x)
model = Model(input, x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

log_file_path = base_path + 'face_attrib_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience/5), verbose=1)
trained_models_path = base_path + 'face_attrib_' + model_name
model_names = trained_models_path + '.{epoch:02d}-{val_loss:.2f}-{loss:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_acc', verbose=1,
                                   save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

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

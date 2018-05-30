import os
import math
import cv2
import numpy as np

def load_lfw_data(image_dir, attributes_path, input_shape = (224, 224, 3), random=True, limit = -1):
  assert os.path.isfile(attributes_path), "Annotation '" + attributes_path + "' not found."
  keys = []
  header = False
  lines = []
  with open(attributes_path) as fi:
    for line in fi:
      if not header:
        header = True
        continue
      if not keys:
        keys = line.rstrip().split('\t')
        continue
      lines.append(line.rstrip().split('\t'))
  labels = keys[3:]

  faces = []
  attributes = []
  raw_attributes = []
  count = 0
  if random:
      # random shuffle
      np.random.shuffle(lines)
  for line in lines:
      image_path = os.path.join(image_dir, ('_'.join(line[0].split())))
      image_path += '_{:04}.jpg'.format(int(line[1]))
      assert os.path.isfile(image_path), '{} not found'.format(image_path)
      face = cv2.imread(image_path)
      face = cv2.resize(face, input_shape[:2])
      faces.append(face.astype('float32'))
      attribute = []
      raw_attribute = []
      for i in range(2, len(line)):
          x = float(line[i])
          if x > 0:
              val = 1
          else:
              val = 0
          attribute.append(val)
          raw_attribute.append(x)
      attributes.append(attribute)
      raw_attributes.append(raw_attribute)
      count += 1
      if limit != -1 and count > limit:
          break

  faces = np.asarray(faces)
  faces = faces / 255.0
  faces = faces - 0.5
  faces = faces * 2.0
  attribtes = np.array(attributes)
  raw_attribtes = np.array(raw_attributes)
  return (faces, attributes, raw_attributes, labels)

def split_data_by_ratio(x, y, split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data

class MacOSFile(object):
    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size

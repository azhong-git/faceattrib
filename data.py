import os
import math
import cv2
import numpy as np
import pickle

def load_attributes(attributes_path):
  keys_lines = pickle.load(open(attributes_path, 'rb'))
  keys = keys_lines['header']
  labels = keys[1:]
  lines = keys_lines['lines']
  multilabel_dict = {}
  for line in lines:
    multilabel_dict[line[0]] = np.array(line[1:])
  return multilabel_dict, labels


def load_data(image_dir, attributes_path, input_shape = (224, 224, 3), crop_middle=True, random=False, limit = -1):
  assert os.path.isfile(attributes_path), "Annotation '" + attributes_path + "' not found."
  keys_lines = pickle.load(open(attributes_path, 'rb'))
  keys = keys_lines['header']
  lines = keys_lines['lines']
  labels = keys[1:]
  if random:
      np.random.shuffle(lines)
  faces = []
  attributes = []
  count = 0
  for line in lines:
      image_path = os.path.join(image_dir, line[0])
      assert os.path.isfile(image_path), '{} not found'.format(image_path)
      bgr_face = cv2.imread(image_path)
      width = bgr_face.shape[1]
      height = bgr_face.shape[0]
      if width != height and crop_middle == True:
        if height > width:
          start = int((height - width) / 2)
          bgr_face = bgr_face[start:(height-start), 0:width, 0:3]
        else:
          start = int((width - height) / 2)
          bgr_face = bgr_face[0:height, start:(width-start), 0:3]
      bgr_face = cv2.resize(bgr_face, input_shape[:2])
      rgb_face = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)
      faces.append(rgb_face.astype('float32'))
      attributes.append(line[1:])
      count += 1
      # print('{} out of {}'.format(count, len(lines)))
      if limit != -1 and count > limit:
          break

  faces = np.asarray(faces)
  faces = faces / 255.0
  faces = faces - 0.5
  faces = faces * 2.0
  attribtes = np.array(attributes)
  return (faces, attributes, labels)

def split_data_by_ratio(x, y, split=[0.8, 0.1, 0.1]):
  sum = 0
  for ratio in split:
    sum += ratio
  assert sum == 1
  num_samples = len(x)
  results = []
  start = 0
  end_ratio = 0
  for i in range(len(split)):
    end_ratio += split[i]
    end_samples = int(end_ratio * num_samples)
    x_ = x[start:end_samples]
    y_ = y[start:end_samples]
    print(start, end_samples)
    results.append([x_, y_])
    start = end_samples
  return results

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

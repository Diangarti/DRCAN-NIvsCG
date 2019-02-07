# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions for downloading and reading MNIST data (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from skimage import color
from multiprocessing import Process
import multiprocessing
import gzip
import skimage as ski
from sklearn.utils import shuffle
import numpy
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin
import cv2
from skimage.transform import resize
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile
from tensorflow.python.util.deprecation import deprecated
from sklearn.preprocessing import LabelEncoder
from skimage import io
import os
import glob
import gc
import numpy as np
# CVDF mirror of http://yann.lecun.com/exdb/mnist/
DEFAULT_SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
def crop_center(img,cropx,cropy):
   # print('to crop')
    y,x,z = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    #print('cropped')    
    return img[starty:starty+cropy,startx:startx+cropx]



def extract_patches(n, cur_path,img, count, config, index, data, labels, label):

    #img = cv2.imread(image_path)
    #print(img.shape)        
    #img = color.rgb2gray(img)
    
    img = crop_center(img, 600,600)
    #print(img.shape)
    x = config.crop_size//240
    for i in range(x):
       # print("i",i)
        for j in range(x):
            count = count+1
            if count < index:
                continue
            data.append(img[(i*64):(((i+1)*240)),(j*240):(((j+1)*240))])
            labels.append(label)
                                
	
                        
            if count == (n+index):
                break
        if count == (n+index):
            break
        
    return data, labels, count
            
            


def load_images_labels(train_dir,config, n, index = 0):
    data_hist = []
    data =[]
    datas = []
	#labels = []
	# path to training dataset
    train_labels = os.listdir(train_dir)

	# encode the labels
    print ("[INFO] encoding labels...")
    le = LabelEncoder()
    le.fit([tl for tl in train_labels])

	# variables to hold features and labels
    features = []
    label1   = []
    labels = []
    

	# loop over all the labels in the folder
    #count = 1
    for i, label in enumerate(train_labels):
        cur_path = train_dir + "/" + label
#        if i > 9:
#            break
        print(i, cur_path)
        count = 0
        
        
        for image_path in glob.glob(cur_path + "/*.jpg"):
            #print(image_path)
            #img = resize((io.imread(image_path)),(640,640))
            count = count + 1
            #if count < index:
            #    continue
            img = io.imread(image_path)
            #img = cv2.imread(image_path)
#            if img.shape[0]<512 or img.shape[1] <512:
#                print("contd")
#                continue;
            #img = color.rgb2gray(img)
            #img = crop_center(img, config.crop_size,config.crop_size)
            data.append(img)
            labels.append(label)
            #df = p.map(dfprocessing_func, [data])[0]
            #data, label, count = extract_patches(n, cur_path, img, count, config, index, data, labels, label)
            #datas.append(data)
            #labels.append(label)
            if count == n:
                break   
            #print("count",count)
            
#        if i == 2:
#            break
#        
#    gc.collect()         
                
    data = np.asarray(data)
    le = LabelEncoder()
    le_labels = le.fit_transform(labels)
	#labels = np.asarray(labels)
    #print(data.shape)
	#labels = dense_to_one_hot(le_labels, 2)
    data = data.reshape(data.shape[0], 240,  240, config.num_channels)
    data, labels = shuffle(data,le_labels)
    #print(data.shape, labels.shape) 
    return data, labels, index



def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


@deprecated(None, 'Please use tf.data to implement this functionality.')
def extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

  Args:
    f: A file object that can be passed into a gzip reader.

  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].

  Raises:
    ValueError: If the bytestream does not start with 2051.

  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


@deprecated(None, 'Please use tf.one_hot on tensors.')
def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


@deprecated(None, 'Please use tf.data to implement this functionality.')
def extract_labels(f, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 numpy array [index].

  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.

  Returns:
    labels: a 1D uint8 numpy array.

  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels


class DataSet(object):
  """Container class for a dataset (deprecated).

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.
  """

  @deprecated(None, 'Please use alternatives such as official/mnist/dataset.py'
              ' from tensorflow/models.')
  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError(
          'Invalid image dtype %r, expected uint8 or float32' % dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 3
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2] * images.shape[3])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate(
          (images_rest_part, images_new_part), axis=0), numpy.concatenate(
              (labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]


@deprecated(None, 'Please use alternatives such as official/mnist/dataset.py'
            ' from tensorflow/models.')
def read_data_sets(train_dir,
                   config,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   seed=None,
                   source_url=DEFAULT_SOURCE_URL):
  train_d = train_dir + "/train" 
  val_d = train_dir + "/val"
  test_d = train_dir + "/val"
  train_images, train_labels, index = load_images_labels(train_dir,config, n =100, index = 0 )
  validation_images, validation_labels, index = load_images_labels(train_dir,config, n = 100, index = 0)
  test_images, test_labels, index = load_images_labels(train_dir,config, n = 40, index = 10)
  #test_images = validation_images 
  #test_labels = validation_labels
#  print("train labels", train_labels.shape)
#  print(len(train_images))
#  validation_size = int(0.2 * (len(train_images)))
#  print(validation_size)
#  test_size = int(0.1 * (len(train_images)))
#  if not 0 <= validation_size <= len(train_images):
#    raise ValueError('Validation size should be between 0 and {}. Received: {}.'
#                     .format(len(train_images), validation_size))
#
#  validation_images = train_images[:validation_size]
#  validation_labels = train_labels[:validation_size]
#  test_images = train_images[validation_size:validation_size+test_size]
#  test_labels = train_labels[validation_size:validation_size+test_size]
#  train_images = train_images[validation_size+test_size:]
#  train_labels = train_labels[validation_size+test_size:]
  print("train_size", train_images.shape[0],"val_size", validation_images.shape[0],"test_size", test_images.shape[0])
  options = dict(dtype=dtype, reshape=reshape, seed=seed)

  train = DataSet(train_images, train_labels, **options)
  validation = DataSet(validation_images, validation_labels, **options)
  test = DataSet(test_images, test_labels, **options)

  return base.Datasets(train=train, validation=validation, test=test)


@deprecated(None, 'Please use alternatives such as official/mnist/dataset.py'
            ' from tensorflow/models.')
def load_mnist(train_dir='MNIST-data'):
  return read_data_sets(train_dir)

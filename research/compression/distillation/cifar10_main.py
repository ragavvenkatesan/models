# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Runs a ResNet model on the CIFAR-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf

import resnet

_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS
# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}

###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
  """Returns a list of filenames."""
  data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

  assert os.path.exists(data_dir), (
      'Run cifar10_download_and_extract.py first to download and extract the '
      'CIFAR-10 data.')

  if is_training:
    return [
        os.path.join(data_dir, 'data_batch_%d.bin' % i)
        for i in range(1, _NUM_DATA_FILES + 1)
    ]
  else:
    return [os.path.join(data_dir, 'test_batch.bin')]


def parse_record(raw_record, is_training):
  """Parse CIFAR-10 image and label from a raw record."""
  # Convert bytes to a vector of uint8 that is record_bytes long.
  record_vector = tf.decode_raw(raw_record, tf.uint8)

  # The first byte represents the label, which we convert from uint8 to int32
  # and then to one-hot.
  label = tf.cast(record_vector[0], tf.int32)
  label = tf.one_hot(label, _NUM_CLASSES)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(record_vector[1:_RECORD_BYTES],
                           [_NUM_CHANNELS, _HEIGHT, _WIDTH])

  # Convert from [depth, height, width] to [height, width, depth], and cast as
  # float32.
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

  image = preprocess_image(image, is_training)

  return image, label


def preprocess_image(image, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, _HEIGHT + 8, _WIDTH + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _NUM_CHANNELS])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  return image


def input_fn(is_training, data_dir, batch_size, num_epochs=1,
             num_parallel_calls=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_parallel_calls: The number of records that are processed in parallel.
      This can be optimized per data set but for generally homogeneous data
      sets, should be approximately the number of available CPU cores.

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = get_filenames(is_training, data_dir)
  dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)

  return resnet.process_record_dataset(dataset, is_training, batch_size,
      _NUM_IMAGES['train'], parse_record, num_epochs, num_parallel_calls)


###############################################################################
# Running the model
###############################################################################
class Cifar10Model(resnet.Model):

  def __init__(self, resnet_size, pool_probes, pool_type, num_probes,
               data_format=None, 
               num_classes=_NUM_CLASSES):
    """These are the parameters that work for CIFAR-10 data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      pool_probes: Number to pool probes by.
      pool_type: either 'max' or 'mean'. 
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
    """
    if resnet_size[0] % 6 != 2:
      raise ValueError('resnet_size must be 6n + 2:', resnet_size[0])

    if resnet_size[1] % 6 != 2:
      raise ValueError('resnet_size must be 6n + 2:', resnet_size[1])

    num_blocks = [ (resnet_size[0] - 2) // 6, (resnet_size[1] - 2) // 6 ]

    super(Cifar10Model, self).__init__(
        resnet_size=resnet_size,
        num_classes=num_classes,
        num_filters=16,
        kernel_size=3,
        conv_stride=1,
        num_probes = num_probes,
        first_pool_size=None,
        first_pool_stride=None,
        second_pool_size=8,
        second_pool_stride=1,
        probe_pool_stride=1,
        probe_pool_size = pool_probes,
        pool_type = pool_type,
        block_fn=resnet.building_block,
        block_sizes=[ [num_blocks[0]] * 3, [num_blocks[1]] * 3 ],
        block_strides=[1, 2, 2],
        final_size=64,
        data_format=data_format)

def cifar10_model_fn(features, labels, mode, params):
  """Model function for CIFAR-10."""
  features = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])

  epochs_1 = params ['train_epochs_mentor']
  learning_rate_fn_mentor = resnet.learning_rate_with_decay_2(
      batch_size=params['batch_size'], batch_denom=128,
      num_images=_NUM_IMAGES['train'], 
      boundary_epochs=[ epochs_1 // 2,
                       3 * epochs_1 // 4,
                       7 * epochs_1 // 8],
      initial_learning_rate = params['initial_learning_rate_mentor'],
      decay_rates=[1, 0.1, 0.01, 0.001])

  epochs_2 = params['train_epochs_mentee']
  learning_rate_fn_mentee = resnet.learning_rate_with_decay_2(
      batch_size=params['batch_size'], batch_denom=128,
      num_images=_NUM_IMAGES['train'], 
      boundary_epochs=[ epochs_1 + epochs_2//4,
                       epochs_1 + 3*epochs_2//4,
                       epochs_1 + 7 * epochs_2//8],
      initial_learning_rate = params['initial_learning_rate_mentee'],
      decay_rates=[1, 0.1, 0.01, 0.001])
  
  epochs_3 = params['train_epochs_finetune']
  learning_rate_fn_finetune = resnet.learning_rate_with_decay_2(
      batch_size=params['batch_size'], batch_denom=128,
      num_images=_NUM_IMAGES['train'], 
      boundary_epochs=[epochs_1 + epochs_2 + epochs_3 //4,
                       epochs_1 + epochs_2 + 3*epochs_3//4,
                       epochs_1 + epochs_2 + 7*epochs_3//8],
      initial_learning_rate = params['initial_learning_rate_finetune'],
      decay_rates=[1, 0.1, 0.01, 0.001])

  # Empirical testing showed that including batch_normalization variables
  # in the calculation of regularized loss helped validation accuracy
  # for the CIFAR-10 dataset, perhaps because the regularization prevents
  # overfitting on the small data set. We therefore include all vars when
  # regularizing and computing loss during training.
  def loss_filter_fn(name):
    return True
  return resnet.resnet_model_fn(features, labels, mode, Cifar10Model,
                                resnet_size=params['resnet_size'],
                                learning_rate_fn_mentor=learning_rate_fn_mentor,
                                learning_rate_fn_mentee=learning_rate_fn_mentee,
                            learning_rate_fn_finetune=learning_rate_fn_finetune,                                
                                momentum=0.9,
                                temperature = params['temperature'],
                                num_probes = params['num_probes'],
                                distillation_coeff=params['distillation_coeff'],
                                weight_decay_coeff=params['weight_decay_coeff'],
                                probes_coeff = params['probes_coeff'],
                                optimizer = params['optimizer'],
                                trainee=params['trainee'],
                                data_format=params['data_format'],
                                pool_probes=params['pool_probes'],
                                pool_type=params['pool_type'],
                                loss_filter_fn=loss_filter_fn)

def main(unused_argv):
  resnet.resnet_main(FLAGS, cifar10_model_fn, input_fn)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)

  parser = resnet.ResnetArgParser()
  # Set defaults that are reasonable for this model.

  parser.set_defaults(data_dir='./cifar10_data',
                      model_dir='./cifar10_model',
                      resnet_size_mentee=1 * 6+2,
                      resnet_size_mentor=1 * 6+2,
                      train_epochs_mentor=100,
                      train_epochs_mentee=200,
                      train_epochs_finetune=100,
                      epochs_per_eval=10,
                      distillation_coeff=0.99,
                      probes_coeff=0.1,
                      temperature=1.5,
                      mentee_optimizer='momentum',
                      mentor_optimizer='momentum',
                      finetune_optimizer='momentum',
                      initial_learning_rate_mentor = 0.03,
                      initial_learning_rate_mentee = 0.03,
                      initial_learning_rate_finetune = 0.003,
                      pool_probes = 2,
                      num_probes = 3,
                      pool_type = 'max',
                      weight_decay_coeff=0.0002,
                      batch_size=500)

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
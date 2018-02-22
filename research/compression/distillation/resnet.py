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
"""Contains definitions for the preactivation form of Residual Networks
(also known as ResNet v2).

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


################################################################################
# Functions for input processing.
################################################################################
def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, num_epochs=1, num_parallel_calls=1):
  """Given a Dataset with raw records, parse each record into images and labels,
  and return an iterator over the records.
  Args:
    dataset: A Dataset representing raw records
    is_training: A boolean denoting whether the input is for training.
    batch_size: The number of samples per batch.
    shuffle_buffer: The buffer size to use when shuffling records. A larger
      value results in better randomness, but smaller values reduce startup
      time and use less memory.
    parse_record_fn: A function that takes a raw record and returns the
      corresponding (image, label) pair.
    num_epochs: The number of epochs to repeat the dataset.
    num_parallel_calls: The number of records that are processed in parallel.
      This can be optimized per data set but for generally homogeneous data
      sets, should be approximately the number of available CPU cores.

  Returns:
    Dataset of (image, label) pairs ready for iteration.
  """
  # We prefetch a batch at a time, This can help smooth out the time taken to
  # load input files as we go through shuffling and processing.
  dataset = dataset.prefetch(buffer_size=batch_size)
  if is_training:
    # Shuffle the records. Note that we shuffle before repeating to ensure
    # that the shuffling respects epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # If we are training over multiple epochs before evaluating, repeat the
  # dataset for the appropriate number of epochs.
  dataset = dataset.repeat(num_epochs)

  # Parse the raw records into images and labels
  dataset = dataset.map(lambda value: parse_record_fn(value, is_training),
                        num_parallel_calls=num_parallel_calls)

  dataset = dataset.batch(batch_size)

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path.
  dataset = dataset.prefetch(1)

  return dataset


################################################################################
# Functions building the ResNet model.
################################################################################
def batch_norm_relu(inputs, training, data_format):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)
  inputs = tf.nn.relu(inputs)
  return inputs


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def building_block(inputs, filters, training, projection_shortcut, strides,
                   data_format):
  """Standard building block for residual networks with BN before convolutions.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, training, data_format)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)

  return inputs + shortcut


def bottleneck_block(inputs, filters, training, projection_shortcut,
                     strides, data_format):
  """Bottleneck block variant for residual networks with BN before convolutions.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first two convolutions. Note
      that the third and final convolution will use 4 times as many filters.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, training, data_format)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)

  return inputs + shortcut


def block_layer(inputs, filters, block_fn, blocks, strides, training, name,
                data_format):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block layer.
  """
  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = 4 * filters if block_fn is bottleneck_block else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                    data_format)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, training, None, 1, data_format)

  return tf.identity(inputs, name)


class Model(object):
  """Base class for building the Resnet v2 Model.
  """

  def __init__(self, resnet_size, num_classes, num_filters, kernel_size,
               conv_stride, first_pool_size, first_pool_stride,
               second_pool_size, second_pool_stride, block_fn, block_sizes,
               block_strides, final_size, data_format=None):
    """Creates a model for classifying an image.

    Args:
      resnet_size: A single integer for the size of the ResNet model.
      num_classes: The number of classes used as labels.
      num_filters: The number of filters to use for the first block layer
        of the model. This number is then doubled for each subsequent block
        layer.
      kernel_size: The kernel size to use for convolution.
      conv_stride: stride size for the initial convolutional layer
      first_pool_size: Pool size to be used for the first pooling layer.
        If none, the first pooling layer is skipped.
      first_pool_stride: stride size for the first pooling layer. Not used
        if first_pool_size is None.
      second_pool_size: Pool size to be used for the second pooling layer.
      second_pool_stride: stride size for the final pooling layer
      block_fn: Which block layer function should be used? Pass in one of
        the two functions defined above: building_block or bottleneck_block
      block_sizes: A list containing n values, where n is the number of sets of
        block layers desired. Each value should be the number of blocks in the
        i-th set.
      block_strides: List of integers representing the desired stride size for
        each of the sets of block layers. Should be same length as block_sizes.
      final_size: The expected size of the model after the second pooling.
      data_format: Input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.
    """
    self.resnet_size = resnet_size

    if not data_format:
      data_format = (
          'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    self.data_format = data_format
    self.num_classes = num_classes
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.conv_stride = conv_stride
    self.first_pool_size = first_pool_size
    self.first_pool_stride = first_pool_stride
    self.second_pool_size = second_pool_size
    self.second_pool_stride = second_pool_stride
    self.block_fn = block_fn
    self.block_sizes = block_sizes
    self.block_strides = block_strides
    self.final_size = final_size

  def __call__(self, inputs, training):
    """Add operations to classify a batch of input images.

    Args:
      inputs: A Tensor representing a batch of input images.
      training: A boolean. Set to True to add operations required only when
        training the classifier.

    Returns:
      A logits Tensor with shape [<batch_size>, self.num_classes].
    """
    with tf.variable_scope('input_transforms'):
      if self.data_format == 'channels_first':
        # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
        # This provides a large performance boost on GPU. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
    with tf.variable_scope('mentor') as scope:
      # mentor
      mentor = conv2d_fixed_padding(
          inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
          strides=self.conv_stride, data_format=self.data_format)
      mentor = tf.identity(mentor, 'mentor_' + 'initial_conv')

      if self.first_pool_size:
        mentor = tf.layers.max_pooling2d(
            inputs=mentor, pool_size=self.first_pool_size,
            strides=self.first_pool_stride, padding='SAME',
            data_format=self.data_format)
        mentor = tf.identity(mentor, 'mentor_' + 'initial_max_pool')
      mentor_probes = []
      for i, num_blocks in enumerate(self.block_sizes[0]):
        num_filters = self.num_filters * (2**i)
        mentor = block_layer(
            inputs=mentor, filters=num_filters, block_fn=self.block_fn,
            blocks=num_blocks, strides=self.block_strides[i],
            training=training, name='mentor_' + 'block_layer{}'.format(i + 1),
            data_format=self.data_format)
        mentor_probes.append(mentor)

      mentor = batch_norm_relu(mentor, training, self.data_format)
      mentor = tf.layers.average_pooling2d(
          inputs=mentor, pool_size=self.second_pool_size,
          strides=self.second_pool_stride, padding='VALID',
          data_format=self.data_format)
      mentor = tf.identity(mentor, 'mentor_' + 'final_avg_pool')

      mentor = tf.reshape(mentor, [-1, self.final_size])
      mentor = tf.layers.dense(inputs=mentor, units=self.num_classes)
      mentor = tf.identity(mentor, 'mentor_' + 'final_dense')

    with tf.variable_scope('mentee') as scope:
      # mentee
      mentee = conv2d_fixed_padding(
          inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
          strides=self.conv_stride, data_format=self.data_format)
      mentee = tf.identity(mentee, 'mentee_' + 'initial_conv')

      if self.first_pool_size:
        mentee = tf.layers.max_pooling2d(
            inputs=mentee, pool_size=self.first_pool_size,
            strides=self.first_pool_stride, padding='SAME',
            data_format=self.data_format)
        mentee = tf.identity(mentee, 'mentee_' + 'initial_max_pool')
      mentee_probes = []
      for i, num_blocks in enumerate(self.block_sizes[1]):
        num_filters = self.num_filters * (2**i)
        mentee = block_layer(
            inputs=mentee, filters=num_filters, block_fn=self.block_fn,
            blocks=num_blocks, strides=self.block_strides[i],
            training=training, name='mentee_' + 'block_layer{}'.format(i + 1),
            data_format=self.data_format)
        mentee_probes.append(mentee)

      mentee = batch_norm_relu(mentee, training, self.data_format)
      mentee = tf.layers.average_pooling2d(
          inputs=mentee, pool_size=self.second_pool_size,
          strides=self.second_pool_stride, padding='VALID',
          data_format=self.data_format)
      mentee = tf.identity(mentee, 'mentee_' + 'final_avg_pool')
      mentee = tf.reshape(mentee, [-1, self.final_size])
      mentee = tf.layers.dense(inputs=mentee, units=self.num_classes)
      mentee = tf.identity(mentee, 'mentee_' + 'final_dense')  

    probe_cost = tf.constant(0.)
    for mentor_feat, mentee_feat in zip(mentor_probes, mentee_probes):
      probe_cost = probe_cost + tf.reduce_sum(tf.losses.mean_squared_error (
                                mentor_feat, mentee_feat))
    return (mentor, mentee, probe_cost)

################################################################################
# Functions for running training/eval/validation loops for the model.
################################################################################
def learning_rate_with_decay(
    batch_size, batch_denom, num_images, boundary_epochs, decay_rates):
  """Get a learning rate that decays step-wise as training progresses.

  Args:
    batch_size: the number of examples processed in each training batch.
    batch_denom: this value will be used to scale the base learning rate.
      `0.1 * batch size` is divided by this number, such that when
      batch_denom == batch_size, the initial learning rate will be 0.1.
    num_images: total number of images that will be used for training.
    boundary_epochs: list of ints representing the epochs at which we
      decay the learning rate.
    decay_rates: list of floats representing the decay rates to be used
      for scaling the learning rate. Should be the same length as
      boundary_epochs.

  Returns:
    Returns a function that takes a single argument - the number of batches
    trained so far (global_step)- and returns the learning rate to be used
    for training the next batch.
  """
  with tf.variable_scope('learning_rate'):
    initial_learning_rate = 0.01 * batch_size / batch_denom
    batches_per_epoch = num_images / batch_size

    # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(global_step):
      global_step = tf.cast(global_step, tf.int32)
      rval = tf.train.piecewise_constant(global_step, boundaries, vals)
      return rval
  return learning_rate_fn

def distillation_coeff_fn(intital_distillation, global_step):
  
  global_step = tf.cast(global_step, tf.int32)
  rval = tf.train.exponential_decay (
                            intital_distillation,
                            global_step, 
                            100000,
                            0.75,
                            staircase = False)
  return rval

def resnet_model_fn(features, labels, mode, model_class, trainee, 
                    distillation_coeff, probes_coeff, resnet_size, num_probes,
                    weight_decay_coeff, learning_rate_fn, momentum, data_format,
                    temperature=1, optimizer='momentum', loss_filter_fn=None):
  """Shared functionality for different resnet model_fns.

  Initializes the ResnetModel representing the model layers
  and uses that model to build the necessary EstimatorSpecs for
  the `mode` in question. For training, this means building losses,
  the optimizer, and the train op that get passed into the EstimatorSpec.
  For evaluation and prediction, the EstimatorSpec is returned without
  a train op, but with the necessary parameters for the given mode.

  Args:
    features: tensor representing input images
    labels: tensor representing class labels for all input images
    mode: current estimator mode; should be one of
      `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
    model_class: a class representing a TensorFlow model that has a __call__
      function. We assume here that this is a subclass of ResnetModel.
    trainee: A string either `'mentee'` or `'mentor`'.
    resnet_size: A list of two integers for the size of the ResNet model for 
      mentor followed by mentee.
    weight_decay_coeff: weight decay rate used to regularize learned variables.
    distillation_coeff: Weight for distillation.
    probes_coeff: weight for probes.
    learning_rate_fn: function that returns the current learning rate given
      the current global_step
    num_probes: How many equally spaced probes do we need. 
    momentum: momentum term used for optimization.
    data_format: Input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
    temperature: A value of temperature to use for distillation. Defaults to 1
      so that it will remain backward compatible.
    loss_filter_fn: function that takes a string variable name and returns
      True if the var should be included in loss calculation, and False
      otherwise. If None, batch_normalization variables will be excluded
      from the loss.
    optimizer: 'adam' and 'momentum' are options.
  Returns:
    EstimatorSpec parameterized according to the input params and the
    current mode.
  """
  with tf.variable_scope('inputs'):
    # Generate a summary node for the images
    tf.summary.image('images', features, max_outputs=6)

  model = model_class(resnet_size, data_format)
  logits_mentor, logits_mentee, probe_cost = model(features, 
                                       mode == tf.estimator.ModeKeys.TRAIN)

  predictions_mentor = {
      'classes': tf.argmax(logits_mentor, axis=1),
      'probabilities': tf.nn.softmax(logits_mentor,  
                       name='softmax_tensor_mentor'),
  }

  predictions_mentee = {
      'classes': tf.argmax(logits_mentee, axis=1),
      'probabilities': tf.nn.softmax(logits_mentee,  
                       name='softmax_tensor_mentee'),
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    if trainee == 'mentor':
      return tf.estimator.EstimatorSpec(mode=mode, 
              predictions=predictions_mentor)
    elif trainee == 'mentee' or trainee == 'finetune':
      return tf.estimator.EstimatorSpec(mode=mode, 
              predictions=predictions_mentee)

  with tf.variable_scope('distillery'):
    temperature_softmax_mentor = tf.nn.softmax((tf.div(logits_mentor, 
                      temperature)), name ='softmax_temperature_tensor_mentor')
    distillation_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                    logits = tf.div(logits_mentee,temperature),
                                    labels = temperature_softmax_mentor))
    probe_scale = probes_coeff * distillation_coeff                                   
    tf.identity(distillation_loss, name='distillation_loss')
    tf.summary.scalar('distillation_loss', distillation_loss)
    tf.summary.scalar('scaled_distillation_loss', distillation_coeff *
                        distillation_loss)
    tf.identity(probe_cost, name='probe_cost')                        
    tf.summary.scalar('probe_loss', probe_cost)
    tf.summary.scalar('scaled_probe_loss', probe_scale *
                        probe_cost)

  with tf.variable_scope('cross_entropy'):
    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy_mentor = tf.losses.softmax_cross_entropy(
        logits=logits_mentor, onehot_labels=labels)
    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy_mentor, name='cross_entropy_mentor')
    tf.summary.scalar('cross_entropy_mentor', cross_entropy_mentor)        

    cross_entropy_mentee = tf.losses.softmax_cross_entropy(
        logits=logits_mentee, onehot_labels=labels)
    tf.identity(cross_entropy_mentee, name='cross_entropy_mentee')
    tf.summary.scalar('cross_entropy_mentee', cross_entropy_mentee)

  # If no loss_filter_fn is passed, assume we want the default behavior,
  # which is that batch_normalization variables are excluded from loss.
  if not loss_filter_fn:
    def loss_filter_fn(name):
      return 'batch_normalization' not in name

  mentor_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope='mentor')
  mentee_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope='mentee')   

  with tf.variable_scope('regularizers'):                                       
    if weight_decay_coeff > 0:
        l2_mentor = weight_decay_coeff * tf.add_n(
            [tf.nn.l2_loss(v) for v in mentor_variables
            if loss_filter_fn(v.name)])
        l2_mentee = weight_decay_coeff * tf.add_n(
            [tf.nn.l2_loss(v) for v in mentee_variables
            if loss_filter_fn(v.name)])          
    else:
      l2_mentor = tf.constant(0.)
      l2_mentee = tf.constant(0.)

  if mode == tf.estimator.ModeKeys.TRAIN:
    with tf.variable_scope('learning_rates'):
      global_step_mentor = tf.train.get_or_create_global_step()
      global_step_mentee = tf.train.get_or_create_global_step()    
      learning_rate_mentor = learning_rate_fn(global_step_mentor)
      learning_rate_mentee = learning_rate_fn(global_step_mentee) * 10
      tf.identity(learning_rate_mentor, name='learning_rate_mentor' )
      tf.summary.scalar('learning_rate_mentor', learning_rate_mentor)
      tf.identity(learning_rate_mentee, name='learning_rate_mentee' )
      tf.summary.scalar('learning_rate_mentee', learning_rate_mentee)

    with tf.variable_scope('mentor_cumulative_loss'):
      # Add weight decay and distillation to the loss.
      loss_mentor = cross_entropy_mentor + weight_decay_coeff * l2_mentor
      tf.summary.scalar('objective', loss_mentor)                                       
      
    with tf.variable_scope('mentee_cumulative_loss'): 
      distillation_coeff_decayed = distillation_coeff_fn(distillation_coeff, 
                                          global_step_mentee) 
      tf.identity(distillation_coeff, name='distillation_coeff_decayed')
      tf.summary.scalar('coeff',distillation_coeff_decayed)                                       
      loss_mentee = cross_entropy_mentee + weight_decay_coeff * l2_mentee + \
                    distillation_coeff_decayed * distillation_loss  + \
                    probe_scale * probe_cost
      tf.summary.scalar('objective', loss_mentee)                                       
                    
    with tf.variable_scope('mentee_finetune'):
      loss_mentee_finetune = cross_entropy_mentee + \
                             weight_decay_coeff * l2_mentee
      tf.summary.scalar('objective', loss_mentee_finetune) 

    if optimizer == 'momentum':
      with tf.variable_scope('mentor_momentum_optimizer'):    
        optimizer_mentor = tf.train.MomentumOptimizer(
          learning_rate=learning_rate_mentor,
          momentum=momentum)
      with tf.variable_scope('mentee_momentum_optimizer'):              
        optimizer_mentee = tf.train.MomentumOptimizer(
          learning_rate=learning_rate_mentee,
          momentum=momentum)
      with tf.variable_scope('finetune_momentum_optimizer'):              
        optimizer_finetune = tf.train.MomentumOptimizer(
          learning_rate=learning_rate_mentee,
          momentum=momentum)

    elif optimizer == 'adam':
      with tf.variable_scope('mentor_adam_optimizer'):         
        optimizer_mentor = tf.train.AdamOptimizer(
          learning_rate=learning_rate_mentor)
      with tf.variable_scope('mentee_adam_optimizer'):              
        optimizer_mentee = tf.train.AdamOptimizer(
          learning_rate=learning_rate_mentee)
      with tf.variable_scope('finetune_adam_optimizer'):              
        optimizer_finetune = tf.train.AdamOptimizer(
          learning_rate=learning_rate_mentee)

    # Batch norm requires update ops to be added as a dependency to train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      with tf.variable_scope('optimizers'):
        train_op_mentor = optimizer_mentor.minimize(loss_mentor, 
                                      global_step_mentor, 
                                      var_list = mentor_variables)
        train_op_mentee = optimizer_mentee.minimize(loss_mentee, 
                                      global_step_mentee, 
                                      var_list = mentee_variables)  
        train_op_finetune = optimizer_finetune.minimize(loss_mentee_finetune, 
                                      global_step_mentee, 
                                      var_list = mentee_variables)                                                                         
  else:
    with tf.variable_scope('mentor_cumulative_loss'):
      # Add weight decay and distillation to the loss.
      loss_mentor = cross_entropy_mentor + weight_decay_coeff * l2_mentor
    with tf.variable_scope('mentee_cumulative_loss'):                                      
      loss_mentee = cross_entropy_mentee + weight_decay_coeff * l2_mentee
    with tf.variable_scope('mentee_finetune'):
      loss_finetune = cross_entropy_mentee + weight_decay_coeff * l2_mentee
    train_op_mentor = None
    train_op_mentee = None
    train_op_finetune = None

  with tf.variable_scope('metrics'):
    accuracy_mentor = tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions_mentor['classes'])
    accuracy_mentee = tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions_mentee['classes'])      
    metrics = {'accuracy_mentor': accuracy_mentor,
              'accuracy_mentee': accuracy_mentee}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy_mentor[1], name='train_accuracy_mentor')
    tf.summary.scalar('train_accuracy_mentor', accuracy_mentor[1])
    tf.identity(accuracy_mentee[1], name='train_accuracy_mentee')
    tf.summary.scalar('train_accuracy_mentee', accuracy_mentee[1])

  saver=tf.train.Saver(var_list = tf.global_variables())

  if trainee == 'mentor':
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions_mentor,
        loss=loss_mentor,
        train_op=train_op_mentor,
        scaffold=tf.train.Scaffold(saver=saver),
        eval_metric_ops=metrics)

  elif trainee == 'mentee':
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions_mentee,
        loss=loss_mentee,
        train_op=train_op_mentee,
        scaffold=tf.train.Scaffold(saver=saver),
        eval_metric_ops=metrics)
  elif trainee == 'finetune':
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions_mentee,
        loss=loss_mentee_finetune,
        train_op=train_op_finetune,
        scaffold=tf.train.Scaffold(saver=saver),
        eval_metric_ops=metrics)    


def resnet_main(flags, model_function, input_function):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  
  # Set up a RunConfig to only save checkpoints once per training cycle.
  run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
  mentor = tf.estimator.Estimator(
      model_fn=model_function, model_dir=flags.model_dir, 
      config=run_config,
      params={
          'resnet_size': [flags.resnet_size_mentor, flags.resnet_size_mentee],
          'data_format': flags.data_format,
          'batch_size': flags.batch_size,
          'distillation_coeff': flags.distillation_coeff,
          'probes_coeff': flags.probes_coeff,
          'weight_decay_coeff': flags.weight_decay_coeff,
          'optimizer': flags.optimizer,
          'temperature': flags.temperature,
          'num_probes': flags.num_probes,
          'trainee': 'mentor'
      })

  for i in range(flags.train_epochs_mentor // flags.epochs_per_eval):
    tensors_to_log = {
        'learning_rate': 'learning_rates/learning_rate_mentor',
        'cross_entropy': 'cross_entropy/cross_entropy_mentor' ,
        'train_accuracy': 'metrics/train_accuracy_mentor'
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    def input_fn_train():
      return input_function(True, flags.data_dir, flags.batch_size,
                            flags.epochs_per_eval, flags.num_parallel_calls)

    print(' *********************** ' )
    print(' Starting a mentor training cycle. [' + str(i) + '/' 
            + str(flags.train_epochs_mentor // flags.epochs_per_eval) + ']')
    print(' *********************** ' )            
    
    mentor.train(input_fn=input_fn_train, hooks=[logging_hook])

    print('Starting to evaluate.')
    # Evaluate the model and print results
    def input_fn_eval():
      return input_function(False, flags.data_dir, flags.batch_size,
                            1, flags.num_parallel_calls)

    eval_results = mentor.evaluate(input_fn=input_fn_eval)
    print(eval_results)

  mentee = tf.estimator.Estimator(
      model_fn=model_function, model_dir=flags.model_dir, 
      config=run_config,
      params={
          'resnet_size': [flags.resnet_size_mentor, flags.resnet_size_mentee],
          'data_format': flags.data_format,
          'batch_size': flags.batch_size,
          'distillation_coeff': flags.distillation_coeff,
          'probes_coeff': flags.probes_coeff,   
          'optimizer': flags.optimizer,
          'weight_decay_coeff': flags.weight_decay_coeff,          
          'temperature': flags.temperature,
          'num_probes': flags.num_probes,                 
          'trainee': 'mentee'
      })

  for i in range(flags.train_epochs_mentee // flags.epochs_per_eval):
    tensors_to_log = {
        'learning_rate': 'learning_rates/learning_rate_mentee',
        'cross_entropy': 'cross_entropy/cross_entropy_mentee',
        'train_accuracy': 'metrics/train_accuracy_mentee',
        'distillation_loss': 'distillery/distillation_loss',
        'distillation_coeff':'mentee_cumulative_loss/distillation_coeff_decayed'
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    def input_fn_train():
      return input_function(True, flags.data_dir, flags.batch_size,
                            flags.epochs_per_eval, flags.num_parallel_calls)

    print(' *********************** ' )
    print(' Starting a mentee training cycle. [' + str(i) + '/' 
            + str(flags.train_epochs_mentee // flags.epochs_per_eval) + ']')
    print(' *********************** ' )

    mentee.train(input_fn=input_fn_train, hooks=[logging_hook])

    print('Starting to evaluate.')
    # Evaluate the model and print results
    def input_fn_eval():
      return input_function(False, flags.data_dir, flags.batch_size,
                            1, flags.num_parallel_calls)

    eval_results = mentee.evaluate(input_fn=input_fn_eval)
    print(eval_results)

  fintune = tf.estimator.Estimator(
      model_fn=model_function, model_dir=flags.model_dir, 
      config=run_config,
      params={
          'resnet_size': [flags.resnet_size_mentor, flags.resnet_size_mentee],
          'data_format': flags.data_format,
          'batch_size': flags.batch_size,
          'distillation_coeff': flags.distillation_coeff,
          'probes_coeff': flags.probes_coeff,   
          'optimizer': flags.optimizer,
          'weight_decay_coeff': flags.weight_decay_coeff,          
          'temperature': flags.temperature,
          'num_probes': flags.num_probes,                 
          'trainee': 'finetune'
      })

  for i in range(flags.train_epochs_finetune // flags.epochs_per_eval):
    tensors_to_log = {
        'learning_rate': 'learning_rates/learning_rate_mentee',
        'cross_entropy': 'cross_entropy/cross_entropy_mentee',
        'train_accuracy': 'metrics/train_accuracy_mentee',
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    def input_fn_train():
      return input_function(True, flags.data_dir, flags.batch_size,
                            flags.epochs_per_eval, flags.num_parallel_calls)

    print(' *********************** ' )
    print(' Starting a mentee finetune cycle. [' + str(i) + '/' 
            + str(flags.train_epochs_finetune // flags.epochs_per_eval) + ']')
    print(' *********************** ' )

    mentee.train(input_fn=input_fn_train, hooks=[logging_hook])

    print('Starting to evaluate.')
    # Evaluate the model and print results
    def input_fn_eval():
      return input_function(False, flags.data_dir, flags.batch_size,
                            1, flags.num_parallel_calls)

    eval_results = finetune.evaluate(input_fn=input_fn_eval)
    print(eval_results)

class ResnetArgParser(argparse.ArgumentParser):
  """Arguments for configuring and running a Resnet Model.
  """

  def __init__(self, resnet_size_choices=None):
    super(ResnetArgParser, self).__init__()
    self.add_argument(
        '--data_dir', type=str, default='./resnet_data',
        help='The directory where the input data is stored.')

    self.add_argument(
        '--num_parallel_calls', type=int, default=5,
        help='The number of records that are processed in parallel '
        'during input processing. This can be optimized per data set but '
        'for generally homogeneous data sets, should be approximately the '
        'number of available CPU cores.')

    self.add_argument(
        '--model_dir', type=str, default='./resnet_model',
        help='The directory where the model will be stored.')

    self.add_argument(
        '--resnet_size_mentor', type=int, default=50,
        choices=resnet_size_choices,
        help='The size of the ResNet Mentor model to use.')

    self.add_argument(
        '--resnet_size_mentee', type=int, default=10,
        choices=resnet_size_choices,
        help='The size of the ResNet Mentee model to use.')

    self.add_argument(
        '--train_epochs_mentor', type=int, default=100,
        help='The number of epochs to use for training.')

    self.add_argument(
        '--train_epochs_mentee', type=int, default=100,
        help='The number of epochs to use for training.')

    self.add_argument(
        '--train_epochs_finetune', type=int, default=100,
        help='The number of epochs to use for training.')

    self.add_argument(
        '--epochs_per_eval', type=int, default=1,
        help='The number of training epochs to run between evaluations.')

    self.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for training and evaluation.')

    self.add_argument(
        '--optimizer', type=str, default='momentum',
        help='Batch size for training and evaluation.')

    self.add_argument(
        '--data_format', type=str, default=None,
        choices=['channels_first', 'channels_last'],
        help='A flag to override the data format used in the model. '
             'channels_first provides a performance boost on GPU but '
             'is not always compatible with CPU. If left unspecified, '
             'the data format will be chosen automatically based on '
             'whether TensorFlow was built for CPU or GPU.')

    self.add_argument(
        '--distillation_coeff', type=float, default=0.01,
        help='Coefficient of distillation to be applied from parent to'
              'child. This is only useful when performing distillaiton.')

    self.add_argument(
        '--probes_coeff', type=float, default=0.0001,
        help='Coefficient of weight to be applied from parent to'
              'child. This is only useful when performing mentoring.')

    self.add_argument(
        '--weight_decay_coeff', type=float, default=0.0002,
        help='Coefficient of weight to be applied from to the'
              'weight decay regularizer.')

    self.add_argument(
        '--temperature', type=float, default=3,
        help='Temperature to be used for the softmax layer')

    self.add_argument(
        '--num_probes', type=int, default=0,
        help='Number of probes to be used')
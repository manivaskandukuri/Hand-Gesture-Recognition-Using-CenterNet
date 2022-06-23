# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=invalid-name
"""DenseNet models for Keras.
Reference:
  - [Densely Connected Convolutional Networks](
      https://arxiv.org/abs/1608.06993) (CVPR 2017)
"""

import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras.layers import BatchNormalization, Activation, Dense, Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Concatenate, Input, ZeroPadding2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras import Model



def dense_block(x, blocks, k, name):
  """A dense block.
  Args:
    x: input tensor.
    blocks: integer, the number of building blocks.
    name: string, block label.
  Returns:
    Output tensor for the block.
  """
  for i in range(blocks):
    x = conv_block(x, k, name=name + '_block' + str(i + 1))
  return x


def transition_block(x, reduction, name):
  """A transition block.
  Args:
    x: input tensor.
    reduction: float, compression rate at transition layers.
    name: string, block label.
  Returns:
    output tensor for the block.
  """
  bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
  x = BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(
          x)
  x = Activation('relu', name=name + '_relu')(x)
  x = Conv2D(
      int(K.int_shape(x)[bn_axis] * reduction),
      1,
      use_bias=False,
      name=name + '_conv')(
          x)
  x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
  return x


def conv_block(x, growth_rate, name):
  """A building block for a dense block.
  Args:
    x: input tensor.
    growth_rate: float, growth rate at dense layers.
    name: string, block label.
  Returns:
    Output tensor for the block.
  """
  bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
  x1 = BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(
          x)
  x1 = Activation('relu', name=name + '_0_relu')(x1)
  x1 = Conv2D(
      4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(
          x1)
  x1 = BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(
          x1)
  x1 = Activation('relu', name=name + '_1_relu')(x1)
  x1 = Conv2D(
      growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(
          x1)
  x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
  return x


def DenseNet(
    blocks,
    k,
    inputs,
    include_top=True,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'):
  """Instantiates the DenseNet architecture.
  Reference:
  - [Densely Connected Convolutional Networks](
      https://arxiv.org/abs/1608.06993) (CVPR 2017)
  This function returns a Keras image classification model,
  optionally loaded with weights pre-trained on ImageNet.
  For image classification use cases, see
  [this page for detailed examples](
    https://keras.io/api/applications/#usage-examples-for-image-classification-models).
  For transfer learning use cases, make sure to read the
  [guide to transfer learning & fine-tuning](
    https://keras.io/guides/transfer_learning/).
  Note: each Keras Application expects a specific kind of input preprocessing.
  For DenseNet, call `tf.keras.applications.densenet.preprocess_input` on your
  inputs before passing them to the model.
  `densenet.preprocess_input` will scale pixels between 0 and 1 and then
  will normalize each channel with respect to the ImageNet dataset statistics.
  Args:
    blocks: numbers of building blocks for the four dense layers.
    include_top: whether to include the fully-connected
      layer at the top of the network.
    weights: one of `None` (random initialization),
      'imagenet' (pre-training on ImageNet),
      or the path to the weights file to be loaded.
    input_tensor: optional Keras tensor
      (i.e. output of `layers.Input()`)
      to use as image input for the model.
    input_shape: optional shape tuple, only to be specified
      if `include_top` is False (otherwise the input shape
      has to be `(224, 224, 3)` (with `'channels_last'` data format)
      or `(3, 224, 224)` (with `'channels_first'` data format).
      It should have exactly 3 inputs channels,
      and width and height should be no smaller than 32.
      E.g. `(200, 200, 3)` would be one valid value.
    pooling: optional pooling mode for feature extraction
      when `include_top` is `False`.
      - `None` means that the output of the model will be
          the 4D tensor output of the
          last convolutional block.
      - `avg` means that global average pooling
          will be applied to the output of the
          last convolutional block, and thus
          the output of the model will be a 2D tensor.
      - `max` means that global max pooling will
          be applied.
    classes: optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified.
    classifier_activation: A `str` or callable. The activation function to use
      on the "top" layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.
      When loading pretrained weights, `classifier_activation` can only
      be `None` or `"softmax"`.
  Returns:
    A `keras.Model` instance.
  """



  bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

  x = ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
  x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
  x = BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(
          x)
  x = Activation('relu', name='conv1/relu')(x)
  x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
  x = MaxPooling2D(3, strides=2, name='pool1')(x)

  x = dense_block(x, blocks[0], k, name='conv2')
  x = transition_block(x, 0.5, name='pool2')
  x = dense_block(x, blocks[1], k, name='conv3')
  x = transition_block(x, 0.5, name='pool3')
  x = dense_block(x, blocks[2], k, name='conv4')
  x = transition_block(x, 0.5, name='pool4')
  x = dense_block(x, blocks[3], k, name='conv5')

  x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
  x = Activation('relu', name='relu')(x)

  if include_top:
    x = GlobalAveragePooling2D(name='avg_pool')(x)

    x = Dense(classes, activation=classifier_activation,
                     name='predictions')(x)
  else:
    if pooling == 'avg':
      x = GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
      x = GlobalMaxPooling2D(name='max_pool')(x)


  # Create model.
  if blocks == [6, 12, 24, 16]:
    model = Model(inputs, x, name='densenet121')
  elif blocks == [6, 12, 32, 32]:
    model = Model(inputs, x, name='densenet169')
  elif blocks == [6, 12, 48, 32]:
    model = Model(inputs, x, name='densenet201')
  elif blocks == [6, 12, 36, 24]:
    model = Model(inputs, x, name='densenet161')
  elif blocks == [6, 12, 64, 48]:
    model = Model(inputs, x, name='densenet264')
  else:
    model = Model(inputs, x, name='densenet')


  return model



def DenseNet121(inputs, 
                include_top=True,
                pooling=None,
                classes=1000):
  """Instantiates the Densenet121 architecture."""
  return DenseNet([6, 12, 24, 16], 32, inputs, include_top, pooling, classes)



def DenseNet169(inputs,
                include_top=True,
                pooling=None,
                classes=1000):
  """Instantiates the Densenet169 architecture."""
  return DenseNet([6, 12, 32, 32], 32, inputs, include_top, pooling, classes)



def DenseNet201(inputs,
                include_top=True,
                pooling=None,
                classes=1000):
  """Instantiates the Densenet201 architecture."""
  return DenseNet([6, 12, 48, 32], 32, inputs, include_top, pooling, classes)



def DenseNet161(inputs,
                include_top=True,
                pooling=None,
                classes=1000):
  """Instantiates the Densenet161 architecture."""
  return DenseNet([6, 12, 36, 24], 48, inputs, include_top, pooling, classes) 



def DenseNet264(inputs,
                include_top=True,
                pooling=None,
                classes=1000):
  """Instantiates the Densenet264 architecture."""
  return DenseNet([6, 12, 64, 48], 32, inputs, include_top, pooling, classes)  

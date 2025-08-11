"""
This module implements mobilenet block.
"""

from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, ReLU


# MobileNet block
def mobilenet_block (x, filters, strides):
    
    x = DepthwiseConv2D(kernel_size = 3, strides = strides, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(filters = filters, kernel_size = 1, strides = 1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x
"""
This module implements mobilenet architecture.
"""

from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, AvgPool2D
from .blocks import mobilenet_block
from tensorflow.keras import Model

class MobileNet(Model):
    """
    Constructs a `keras.models.Model` object using the given block count.

    :param inputs: input tensor

    :param block: a mobilenet block

    :param include_top: if true, includes classification layers

    :param classes: number of classes to classify (include_top must be true)

    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)


    """
    def __init__(
        self,
        inputs,
        block=mobilenet_block,
        include_top=True,
        classes=1000,
        *args,
        **kwargs
    ):

        x = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = block(x, filters = 64, strides = 1)
        x = block(x, filters = 128, strides = 2)
        x = block(x, filters = 128, strides = 1)
        x = block(x, filters = 256, strides = 2)
        x = block(x, filters = 256, strides = 1)
        x = block(x, filters = 512, strides = 2)
        for _ in range (5):
            x = block(x, filters = 512, strides = 1)
        x = block(x, filters = 1024, strides = 2)
        x = block(x, filters = 1024, strides = 1)

        if include_top:
            assert classes > 0

            x = AvgPool2D (pool_size = 7, strides = 1, data_format='channels_first')(x)
            x = Dense (units = 1000, activation = 'softmax')(x)

            super(MobileNet, self).__init__(inputs=inputs, outputs=x, *args, **kwargs)
        else:
            # Else output each stages features
            super(MobileNet, self).__init__(inputs=inputs, outputs=x, *args, **kwargs)

'''# MobileNet block
def mobilnet (input, include_top=True):
    x = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = mobilenet_block(x, filters = 64, strides = 1)
    x = mobilenet_block(x, filters = 128, strides = 2)
    x = mobilenet_block(x, filters = 128, strides = 1)
    x = mobilenet_block(x, filters = 256, strides = 2)
    x = mobilenet_block(x, filters = 256, strides = 1)
    x = mobilenet_block(x, filters = 512, strides = 2)
    for _ in range (5):
        x = mobilenet_block(x, filters = 512, strides = 1)
    x = mobilenet_block(x, filters = 1024, strides = 2)
    x = mobilenet_block(x, filters = 1024, strides = 1)
    x = AvgPool2D (pool_size = 7, strides = 1, data_format='channels_first')(x)
    if (include_top):
      x = Dense (units = 1000, activation = 'softmax')(x)
    

    return x'''
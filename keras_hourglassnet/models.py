'''
This module implements popular two-dimensional residual models.
'''

from tensorflow.keras.layers import Conv2D, Input, Activation, BatchNormalization, Add 
from tensorflow.keras.layers import UpSampling2D, ZeroPadding2D, GlobalAveragePooling2D, Dense
from tensorflow.keras import Model

from keras_hourglassnet.blocks import pre, hourglass_module, residual



class HourglassNet(Model):
    """
    Arguments
      num_stacks: number of hourglass modules.
      cnv_dim: number of filters after the resolution is decreased.
      inres: network input shape, should be a multiple of 128.
      weights: one of `None` (random initialization),
            'ctdet_coco' (pre-training on COCO for 2D object detection),
            'hpdet_coco' (pre-training on COCO for human pose detection),
            or the path to the weights file to be loaded.
      dims: numbers of channels in the hourglass blocks.
    """

    def __init__(
        self,
        inputs,
        num_stacks, 
        cnv_dim=256, 
        inres=(512, 512),
        dims=[256, 384, 384, 384, 512],
        include_top=True,
        classes=1000,
        *args,
        **kwargs
    ):

        inter = pre(inputs, cnv_dim)
        prev_inter = None

        for i in range(num_stacks):
          prev_inter = inter
          inter = hourglass_module(inter, cnv_dim, i, dims)
          if i < num_stacks - 1:
            inter_ = Conv2D(cnv_dim, 1, use_bias=False, name='inter_.%d.0' % i)(prev_inter)
            inter_ = BatchNormalization(epsilon=1e-5, name='inter_.%d.1' % i)(inter_)

            cnv_ = Conv2D(cnv_dim, 1, use_bias=False, name='cnv_.%d.0' % i)(inter)
            cnv_ = BatchNormalization(epsilon=1e-5, name='cnv_.%d.1' % i)(cnv_)

            inter = Add(name='inters.%d.inters.add' % i)([inter_, cnv_])
            inter = Activation('relu', name='inters.%d.inters.relu' % i)(inter)
            inter = residual(inter, cnv_dim, 'inters.%d' % i)

        if include_top:
            assert classes > 0

            x = GlobalAveragePooling2D(name="pool5")(x)
            x = Dense(classes, activation="softmax", name="fc1000")(x)

            super(HourglassNet, self).__init__(inputs=inputs, outputs=x, *args, **kwargs)
        else:
            # Else output each stages features
            super(HourglassNet, self).__init__(inputs=inputs, outputs=inter, *args, **kwargs)
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, SeparableConv2D
from deformable_conv.deform_conv import tf_batch_map_offsets


class Offset2D(object):
    """Offset Conv2D"""

    def __init__(self, filters, use_separate_conv=True, **kwargs):
        """Init"""

        self.filters = filters
        if use_separate_conv:
            self.offset_conv = SeparableConv2D(filters=filters * 2, kernel_size=(3, 3), padding='same',
                                                   use_bias=False)
        else:
            self.offset_conv = Conv2D(filters=filters*2, kernel_size=(3, 3), padding='same',
                                                   use_bias=False)

        

    def __call__(self, x):
        # TODO offsets probably have no nonlinearity?
        x_shape = tf.shape(x)
        offsets = self.offset_conv(x)
        offsets = self._to_bc_h_w_2(offsets, x_shape)
        x = self._to_bc_h_w(x, x_shape)
        x_offset = tf_batch_map_offsets(x, offsets)
        x_offset = self._to_b_h_w_c(x_offset, x_shape)
        return x_offset

    def compute_output_shape(self, input_shape):
        return input_shape

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, h, w, 2c) -> (b*c, h, w, 2)"""
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, (-1, x_shape[1], x_shape[2], 2))
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, h, w, c) -> (b*c, h, w)"""
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, (-1, x_shape[1], x_shape[2]))
        return x

    @staticmethod
    def _to_b_h_w_c(x, x_shape):
        """(b*c, h, w) -> (b, h, w, c)"""
        x = tf.reshape(
            x, (-1, x_shape[3], x_shape[1], x_shape[2])
        )
        x = tf.transpose(x, [0, 2, 3, 1])
        return x
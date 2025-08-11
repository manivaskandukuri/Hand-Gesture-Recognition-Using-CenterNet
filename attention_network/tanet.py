import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Activation, Permute
from tensorflow.keras.activations import sigmoid


class BasicConv(Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1,1),
        padding='same',
        data_format='channels_first',
        activation=False,
        bn=True,
        use_bias=False,
    ):
        super(BasicConv, self).__init__()
        self.filters = filters
        self.conv = Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            use_bias=use_bias,
        )
        self.bn = (
            BatchNormalization(axis=1)
            if bn
            else None
        )
        self.activation = Activation('relu') if activation else None

    def call(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ZPool(Layer):
    def __init__(self):
      super(ZPool, self).__init__()
    def call(self, x):
        return tf.concat( [tf.expand_dims(tf.reduce_max(x, axis=1), axis=1),
                tf.expand_dims(tf.reduce_mean(x, axis=1), axis=1)], axis=1)



class AttentionGate(Layer):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(
            1, kernel_size, strides=1, padding='same', 
            activation=False,
        )

    def call(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = sigmoid(x_out)
        return x * scale


class TripletAttention(Layer):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        self.hw = AttentionGate() if not no_spatial else None

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'no_spatial': self.no_spatial,
        })
        return config

    def call(self, x):
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        x_perm1 = tf.transpose(x, perm=[0, 2, 1, 3])
        x_out1 = self.cw(x_perm1)
        x_out11 = tf.transpose(x_out1, perm=[0, 2, 1, 3])
        x_perm2 = tf.transpose(x, perm=[0, 3, 2, 1])
        x_out2 = self.hc(x_perm2)
        x_out21 = tf.transpose(x_out2, perm=[0, 3, 2, 1])
        
        if not self.no_spatial:
          x_out = self.hw(x)
          x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
          x_out = 1 / 2 * (x_out11 + x_out21)

        x_out = tf.transpose(x_out, perm=[0, 2, 3, 1])
        return x_out
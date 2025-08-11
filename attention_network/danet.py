import tensorflow as tf
from tensorflow.keras import backend as K

class PAM_Module(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(PAM_Module, self).__init__()

        self.filters = filters

        axis = 3 if K.image_data_format() == "channels_last" else 1
        self.concat = tf.keras.layers.Concatenate(axis=axis)

        self.conv1x1_bn_relu_1 = tf.keras.layers.Conv2D(filters, (1, 1), strides=(1, 1), padding="same")
        self.conv1x1_bn_relu_2 = tf.keras.layers.Conv2D(filters, (1, 1), strides=(1, 1), padding="same")
        self.conv1x1_bn_relu_3 = tf.keras.layers.Conv2D(filters, (1, 1), strides=(1, 1), padding="same")

        self.gamma = None

        self.softmax = tf.keras.layers.Activation("softmax")

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
        })
        return config
    
    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(1,),
            initializer="random_normal",
            name="pam_gamma",
            trainable=True,
            )
    
    def call(self, x, training=None):
        BS, H, W, C = x.shape
        query = self.conv1x1_bn_relu_1(x, training=training)
        key = self.conv1x1_bn_relu_2(x, training=training)
        if K.image_data_format() == "channels_last":
            query = tf.keras.layers.Reshape((H * W, -1))(query) 
            key = tf.keras.layers.Reshape((H * W, -1))(key)

            energy = tf.linalg.matmul(query, key, transpose_b=True)
        else:
            query = tf.keras.layers.Reshape((-1, H * W))(query)
            key = tf.keras.layers.Reshape((-1, H * W))(key)
            
            energy = tf.linalg.matmul(query, key, transpose_a=True)
        
        attention = self.softmax(energy)
        value = self.conv1x1_bn_relu_3(x, training=training)
        if K.image_data_format() == "channels_last":
            value = tf.keras.layers.Reshape((H * W, -1))(value) 
            out = tf.transpose(tf.linalg.matmul(value, attention, transpose_a=True), perm=[0,2,1])
            #print("out shape before:", out.shape)
        else:
            value = tf.keras.layers.Reshape((-1, H * W))(value)
            out = tf.linalg.matmul(value, attention)

        out = tf.keras.layers.Reshape(x.shape[1:])(out)
        out = self.gamma * out + x

        return out

class CAM_Module(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(CAM_Module, self).__init__()

        self.filters = filters

        self.gamma = None

        self.softmax = tf.keras.layers.Activation("softmax")

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
        })
        return config
    
    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(1,),
            initializer="random_normal",
            name="cam_gamma",
            trainable=True,
            )
    
    def call(self, x, training=None):
        BS, H, W, C = x.shape

        if K.image_data_format() == "channels_last":
            query = tf.keras.layers.Reshape((-1, C))(x) 
            key = tf.keras.layers.Reshape((-1, C))(x)

            energy = tf.linalg.matmul(query, key, transpose_a=True)
            energy_2 = tf.math.reduce_max(energy, axis=1, keepdims=True)[0] - energy
        else:
            query = tf.keras.layers.Reshape((C, -1))(query)
            key = tf.keras.layers.Reshape((C, -1))(key)
        
            energy = tf.linalg.matmul(query, key, transpose_b=True)
            energy_2 = tf.math.reduce_max(energy, axis=-1, keepdims=True)[0] - energy
        
        attention = self.softmax(energy_2)

        if K.image_data_format() == "channels_last":
            value = tf.keras.layers.Reshape((-1, C))(x)
            out = tf.transpose(tf.linalg.matmul(attention, value, transpose_b=True), perm=[0,2,1]) 
        else:
            value = tf.keras.layers.Reshape((C, -1))(x)
            out = tf.linalg.matmul(attention, value)

        out = tf.keras.layers.Reshape(x.shape[1:])(out)
        out = self.gamma * out + x

        return out

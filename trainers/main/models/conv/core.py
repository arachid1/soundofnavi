import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, AveragePooling2D
import numpy as np
from tensorflow.python.keras import backend


class InvertedResidual(Layer):
    def __init__(self, filters, strides, activation=ReLU(), kernel_size=3, expansion_factor=6,
                 regularizer=None, trainable=True, name=None, **kwargs):
        super(InvertedResidual, self).__init__(
            trainable=trainable, name=name, **kwargs)
        self.filters = filters
        self.strides = strides
        self.kernel_size = kernel_size
        self.expansion_factor = expansion_factor
        self.activation = activation
        self.regularizer = regularizer
        self.channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    def build(self, input_shape):
        input_channels = int(input_shape[self.channel_axis])  # C
        self.ptwise_conv1 = Conv2D(filters=int(input_channels*self.expansion_factor),
                                   kernel_size=1, kernel_regularizer=self.regularizer, use_bias=False)
        self.dwise = DepthwiseConv2D(kernel_size=self.kernel_size, strides=self.strides,
                                     kernel_regularizer=self.regularizer, padding='same', use_bias=False)
        self.ptwise_conv2 = Conv2D(filters=self.filters, kernel_size=1,
                                   kernel_regularizer=self.regularizer, use_bias=False)
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()
        
        # input = C 
        # 1x1 -> 4C
        # depthwise 3x3 -> 4C
        # 1x1 -> C 
        # ouput = C

    def call(self, input_x, training=False):
        # Expansion
        x = self.ptwise_conv1(input_x)
        x = self.bn1(x, training=training)
        x = self.activation(x)
        # Spatial filtering
        x = self.dwise(x)
        x = self.bn2(x, training=training)
        x = self.activation(x)
        # back to low-channels w/o activation
        x = self.ptwise_conv2(x)
        x = self.bn3(x, training=training)
        # Residual connection only if i/o have same spatial and depth dims
        if input_x.shape[1:] == x.shape[1:]:
            x += input_x
        return x

    def get_config(self):
        cfg = super(InvertedResidual, self).get_config()
        cfg.update({'filters': self.filters,
                    'strides': self.strides,
                    'regularizer': self.strides,
                    'expansion_factor': self.expansion_factor,
                    'activation': self.activation})
        return cfg

def create_bn_act_1_conv_pool(INPUT, KERNEL_SIZE, POOL_SIZE, CHANNELS, PADDING):
    x = BatchNormalization()(INPUT)
    x = ReLU()(x)
    x = Conv2D(CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING, activation="relu")(x)
    x = AveragePooling2D(pool_size=POOL_SIZE, padding=PADDING)(x)
    return x

def create_bn_act_3_convs_pool(INPUT, KERNEL_SIZE, POOL_SIZE, CHANNELS, PADDING):
    x = BatchNormalization()(INPUT)
    x = ReLU()(x)
    x = Conv2D(CHANNELS, kernel_size=1, padding=PADDING, activation="relu")(x)
    x = Conv2D(CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING, activation="relu")(x)
    x = Conv2D(CHANNELS, kernel_size=1, padding=PADDING, activation="relu")(x)
    x = AveragePooling2D(pool_size=POOL_SIZE, padding=PADDING)(x)
    return x

def create_bn_act_3_strided_convs(INPUT, KERNEL_SIZE, POOL_SIZE, CHANNELS, PADDING):
    x = BatchNormalization()(INPUT)
    x = ReLU()(x)
    x = Conv2D(CHANNELS, kernel_size=1, padding=PADDING, activation="relu")(x)
    x = Conv2D(CHANNELS, kernel_size=KERNEL_SIZE, strides=2, padding=PADDING, activation="relu")(x)
    x = Conv2D(CHANNELS, kernel_size=1, padding=PADDING, activation="relu")(x)
    x = AveragePooling2D(pool_size=POOL_SIZE, padding=PADDING)(x)
    return x
    
def create_3_convs_relu_avg_bn(INPUT, KERNEL_SIZE, POOL_SIZE, CHANNELS, PADDING):
    x = Conv2D(CHANNELS, kernel_size=1, padding=PADDING, activation="relu")(INPUT)
    x = Conv2D(CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING, activation="relu")(x)
    x = Conv2D(CHANNELS, kernel_size=1, padding=PADDING, activation="relu")(x)
    x = AveragePooling2D(pool_size=POOL_SIZE, padding=PADDING)(x)
    x = BatchNormalization()(x)
    return x
    
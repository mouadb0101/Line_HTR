import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.regularizers import l2

from config import *


class MyConv2D(tf.keras.Model):
    def __init__(self, filtres, kernel, strides=(1, 1), padding='valid', batchNorma=False):
        super(MyConv2D, self).__init__()
        self.conv = layers.Conv2D(filtres, kernel, strides, padding, kernel_initializer='he_normal', use_bias=False)
        self.bn = layers.BatchNormalization(momentum=0.1, epsilon=1e-05)
        self.relu = layers.Activation('relu')  # layers.LeakyReLU()
        self.batchNorma = batchNorma

    def call(self, inputs):
        x = self.conv(inputs)
        if self.batchNorma:
            x = self.bn(x)
        return self.relu(x)


class VGG12(tf.keras.Model):
    def __init__(self):
        super(VGG12, self).__init__()
        # self.sin = Sin(1.0)
        self.conv1 = MyConv2D(filtres=64, kernel=(3, 3), strides=(1, 1), padding='same',
                              batchNorma=False)  # (None, 800, 64, 64)
        self.max1 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')  # (None, 400, 32, 64)
        self.conv2 = MyConv2D(filtres=128, kernel=(3, 3), strides=(1, 1), padding='same',
                              batchNorma=False)  # (None, 400, 32, 128)
        self.max2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.conv3 = MyConv2D(filtres=256, kernel=(3, 3), strides=(1, 1), padding='same',
                              batchNorma=False)  # (None, 200, 16, 256)
        self.conv4 = MyConv2D(filtres=256, kernel=(3, 3), strides=(1, 1), padding='same',
                              batchNorma=True)  # (None, 200, 16, 256)

        self.max3 = layers.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding='same')
        self.conv5 = MyConv2D(filtres=512, kernel=(3, 3), strides=(1, 1), padding='same',
                              batchNorma=False)  # (None, 100, 8, 512)
        self.conv6 = MyConv2D(filtres=512, kernel=(3, 3), strides=(1, 1), padding='same',
                              batchNorma=True)  # (None, 100, 8, 512)

        self.max4 = layers.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding='same')
        self.conv7 = MyConv2D(filtres=512, kernel=(1, 1), strides=(1, 1), padding='valid',
                              batchNorma=True)  # (None, 100, 2, 512)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.max1(x)
        x = self.conv2(x)
        m2 = x = self.max2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        m3 = x = self.max3(x)
        x = self.conv5(x)
        x = self.conv6(x)
        m4 = x = self.max4(x)
        x = self.conv7(x)

        return x


class gMLPLayer(layers.Layer):
    def __init__(self, num_patches, embedding_dim, dropout_rate, *args, **kwargs):
        super(gMLPLayer, self).__init__(*args, **kwargs)

        self.channel_projection1 = keras.Sequential(
            [
                layers.Dense(units=embedding_dim * 2),
                tfa.layers.GELU(),
                layers.Dropout(rate=dropout_rate),
            ]
        )

        self.channel_projection2 = layers.Dense(units=embedding_dim)

        self.spatial_projection = layers.Dense(
            units=num_patches, bias_initializer="Ones"
        )

        self.normalize1 = layers.LayerNormalization(epsilon=1e-6)
        self.normalize2 = layers.LayerNormalization(epsilon=1e-6)

    def spatial_gating_unit(self, x):
        # Split x along the channel dimensions.
        # Tensors u and v will in th shape of [batch_size, num_patchs, embedding_dim].
        u, v = tf.split(x, num_or_size_splits=2, axis=2)
        # Apply layer normalization.
        v = self.normalize2(v)
        # Apply spatial projection.
        v_channels = tf.linalg.matrix_transpose(v)
        v_projected = self.spatial_projection(v_channels)
        v_projected = tf.linalg.matrix_transpose(v_projected)
        # Apply element-wise multiplication.
        return u * v_projected

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize1(inputs)
        # Apply the first channel projection. x_projected shape: [batch_size, num_patches, embedding_dim * 2].
        x_projected = self.channel_projection1(x)
        # Apply the spatial gating unit. x_spatial shape: [batch_size, num_patches, embedding_dim].
        x_spatial = self.spatial_gating_unit(x_projected)
        # Apply the second channel projection. x_projected shape: [batch_size, num_patches, embedding_dim].
        x_projected = self.channel_projection2(x_spatial)
        # Add skip connection.
        return x + x_projected


def get_model():
    input_shape = (img_w, img_h, 1)
    inputs = layers.Input(shape=input_shape, name='the_input', dtype='float32')

    vgg = VGG12()
    x_out = vgg(inputs)

    shape = x_out.get_shape()
    x = layers.AveragePooling2D(pool_size=(1, shape[2]), strides=(1, shape[2]))(x_out)
    x = tf.squeeze(x, axis=2)

    blocks = keras.Sequential(
        [gMLPLayer(maxTextLen, gMlp_units, gMLP_dropout) for _ in range(L)]
    )
    x = blocks(x)

    if not use_gMlp_only:
        if use_lstm:
            layer1 = layers.LSTM(units=rnn_units, return_sequences=True, kernel_regularizer=l2(0.0000001),
                                 activity_regularizer=l2(0.0000001))
            layer2 = layers.LSTM(units=rnn_units, return_sequences=True, kernel_regularizer=l2(0.0000001),
                                 activity_regularizer=l2(0.0000001))
        else:
            layer1 = layers.GRU(units=rnn_units, return_sequences=True, kernel_regularizer=l2(0.0000001),
                                activity_regularizer=l2(0.0000001))
            layer2 = layers.GRU(units=rnn_units, return_sequences=True, kernel_regularizer=l2(0.0000001),
                                activity_regularizer=l2(0.0000001))
        x = layers.Bidirectional(layer1)(x)
        x = layers.Bidirectional(layer2)(x)

    output = layers.Dense(num_classes, activation='softmax', name="softmax")(x)
    model = keras.models.Model(inputs=inputs, outputs=[output])
    model.summary()
    return model


if __name__ == "__main__":
    model = get_model()

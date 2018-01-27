# -*- coding: utf-8 -*-
# python 2
'''
U-net 3d model을 구성하는 소스코드이다. 이 model은 중심 축을 기준으로 좌우 대칭의 구조를 갖고있다. 이러한 구조는
model에 input되는 이미지의 축소, 확장을 통해 data가 갖고있는 context, precise localization을 진행한다.
'''
import numpy as np

from keras import backend as K
from keras.layers import Input, Flatten, Dense, Dropout, Activation
from keras.layers import concatenate, Add, Lambda
from keras.layers import Conv3D, Conv3DTranspose, MaxPooling3D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

from metrics import dice_coefficient_loss, dice_coefficient

K.set_image_data_format('channels_first')

def u_net3d_model(dropout_rate = 0.5,
                  optimizer = Adam,
                  initial_learning_rate = 5e-4,
                  output_weight = [0.34, 0.33, 0.33],
                  loss_function = 'sparse_categorical_crossentropy',
                  metrics_function = ['sparse_categorical_accuracy'],
                  input_shape = [4, 33, 33, 33],
                  convolution_kernel = (3, 3, 3),
                  convolution_strides = (1, 1, 1),
                  maxpooling_kernel = (2, 2, 2),
                  maxpooling_strides = (2, 2, 2),
                  deconvolution_kernel = (3, 3, 3),
                  deconvolution_strides = (2, 2, 2)):

    inputs = Input(input_shape) # (4, 33, 33, 33)

    conv1 = convolution3D(input_layer = inputs, output_filters = 16, kernel = convolution_kernel, strides = convolution_strides, padding = 'same', activation = 'relu') # (16, 33, 33, 33)
    conv2 = convolution3D(input_layer = conv1, output_filters = 32, kernel = convolution_kernel, strides = convolution_strides, padding = 'same', activation = 'relu') # (32, 33, 33, 33)

    maxpool1 = maxpooling3D(input_layer = conv2, pool_size = maxpooling_kernel, strides = maxpooling_strides, padding = 'valid') # (32, 16, 16, 16)

    conv3 = convolution3D(input_layer = maxpool1, output_filters = 64, kernel = convolution_kernel, strides = convolution_strides, padding = 'same', activation = 'relu') # (64, 16, 16, 16)
    conv4 = convolution3D(input_layer = conv3, output_filters = 128, kernel = convolution_kernel, strides = convolution_strides, padding = 'same', activation = 'relu') # (128, 16, 16, 16)

    maxpool2 = maxpooling3D(input_layer = conv4, pool_size = maxpooling_kernel, strides = maxpooling_strides, padding = 'valid') # (128, 8, 8, 8)

    conv5 = convolution3D(input_layer = maxpool2, output_filters = 256, kernel = convolution_kernel, strides = convolution_strides, padding = 'same', activation = 'relu') # (256, 8, 8, 8)
    conv6 = convolution3D(input_layer = conv5, output_filters = 512, kernel = convolution_kernel, strides = convolution_strides, padding = 'same', activation = 'relu') # (512, 8, 8, 8)
    conv7 = convolution3D(input_layer = conv6, output_filters = 128, kernel = convolution_kernel, strides = convolution_strides, padding = 'same', activation = 'relu') # (128, 8, 8, 8)

    deconv1 = deconvolution3D(input_layer = conv7, output_filters = 128, kernel = deconvolution_kernel, strides = deconvolution_strides, padding = 'same') # (128, 16, 16, 16)

    merge1 = residual_connection(deconv1, conv4) # (128 + 128, 16, 16, 16)

    conv8 = convolution3D(input_layer = merge1, output_filters = 64, kernel = convolution_kernel, strides = convolution_strides, padding = 'same', activation = 'relu') # (64, 16, 16, 16)
    conv9 = convolution3D(input_layer = conv8, output_filters = 32, kernel = convolution_kernel, strides = convolution_strides, padding = 'same', activation = 'relu') # (32, 16, 16, 16)

    deconv2 = deconvolution3D(input_layer = conv9, output_filters = 32, kernel = deconvolution_kernel, strides = deconvolution_strides, padding = 'valid') # (32, 33, 33, 33)

    merge2 = residual_connection(deconv2, conv2) # (32 + 32, 33, 33, 33)

    conv10 = convolution3D(input_layer = merge2, output_filters = 16, kernel = convolution_kernel, strides = convolution_strides, padding = 'same', activation = 'relu') # (16, 33, 33, 33)
    conv11 = convolution3D(input_layer = conv10, output_filters = 8, kernel = convolution_kernel, strides = convolution_strides, padding = 'same', activation = 'relu') # (8, 33, 33, 33)

    last_conv1 = last_convolution3D(conv11) # (8,  33, 33, 33) to (1, 33, 33, 33)
    last_conv2 = last_convolution3D(conv9)  # (32, 16, 16, 16) to (1, 16, 16, 16)
    last_conv3 = last_convolution3D(conv7)  # (128, 8,  8,  8) to (1,  8,  8,  8)

    flattened1 = Flatten()(last_conv1) # (None, 35937)
    flattened2 = Flatten()(last_conv2) # (None, 4096)
    flattened3 = Flatten()(last_conv3) # (None, 512)

    fc1 = fully_connected(flattened1, output_nodes = 125, activation = 'relu')
    fc2 = fully_connected(flattened2, output_nodes = 125, activation = 'relu')
    fc3 = fully_connected(flattened3, output_nodes = 125, activation = 'relu')

    drop1 = Dropout(rate = dropout_rate)(fc1)
    drop2 = Dropout(rate = dropout_rate)(fc2)
    drop3 = Dropout(rate = dropout_rate)(fc3)

    fc4 = fully_connected(drop1, output_nodes = 5)
    fc5 = fully_connected(drop2, output_nodes = 5)
    fc6 = fully_connected(drop3, output_nodes = 5)

    # drop4 = Dropout(rate = dropout_rate)(fc4)
    # drop5 = Dropout(rate = dropout_rate)(fc5)
    # drop6 = Dropout(rate = dropout_rate)(fc6)

    soft1 = Activation('softmax')(fc4)
    soft2 = Activation('softmax')(fc5)
    soft3 = Activation('softmax')(fc6)

    multi1 = multiply(soft1, output_weight[0])
    multi2 = multiply(soft2, output_weight[1])
    multi3 = multiply(soft3, output_weight[2])

    add1 = add(multi1, multi2)
    outputs = add(add1, multi3)

    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizer(lr = initial_learning_rate), loss = loss_function, metrics = metrics_function)

    return model

def fully_connected(input_layer, output_nodes, activation = None):
    if activation == None:
        layer = Dense(units = output_nodes)(input_layer)
        layer = BatchNormalization()(layer)
    else:
        layer = Dense(units = output_nodes)(input_layer)
        layer = BatchNormalization()(layer)
        layer = Activation(activation)(layer)
    return layer

def last_convolution3D(input_layer, output_filter = 1, kernel = (1, 1, 1), strides = (1, 1, 1), padding = 'same'):
    layer = Conv3D(output_filter, kernel_size = kernel, padding = padding,
                   strides = strides, data_format = 'channels_first')(input_layer)
    return layer

def convolution3D(input_layer, output_filters, kernel, strides, padding, activation):
    layer = Conv3D(output_filters, kernel_size = kernel, padding = padding,
                   strides = strides, data_format = 'channels_first')(input_layer)
    layer = BatchNormalization()(layer)
    layer = Activation(activation)(layer)
    return layer

def maxpooling3D(input_layer, pool_size, strides, padding):
    layer = MaxPooling3D(pool_size = pool_size, strides = strides, padding = padding, data_format = 'channels_first')(input_layer)
    return layer

def deconvolution3D(input_layer, output_filters, kernel, strides, padding):
    layer = Conv3DTranspose(filters = output_filters, kernel_size = kernel, strides = strides,
                            padding = padding, data_format = 'channels_first')(input_layer)
    return layer

def multiply(input_layer, scalar):
    layer = Lambda(lambda x : x * scalar)(input_layer)
    return layer

def add(layer1, layer2):
    layer = Add()([layer1, layer2])
    return layer

def residual_connection(deconv_layer, conv_layer):
    residual = concatenate([deconv_layer, conv_layer], axis = 1)
    return residual

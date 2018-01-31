# -*- coding: utf-8 -*-
# python 2
# Brats2015 data를 33x33x33의 3차원 이미지로 슬라이스된 이미지 데이터를 3D U-net구조의 CNN model에 학습하는 과정.
from __future__ import print_function

import os
import time
import numpy as np

from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau,TensorBoard
from keras.optimizers import Adam

from metrics import dice_coefficient, dice_coefficient_loss
from u_net3d_model import u_net3d_model

output_path = input('INPUT OUTPUT PATH : ')
model_dir_name = 'model/'
os.makedirs(output_path + model_dir_name)
model_path = output_path + model_dir_name

train_patch_name = output_path + 'HGG_train.pat'
label_patch_name = output_path + 'HGG_label.pat'
check_point = model_path + 'model_check_point.{epoch:02d}-{val_acc:.4f}.hdf5'
log_file = model_path + 'training.log'
model_file = model_path + 'u_net3d_model.h5'
tensorboard_log_file = model_path + 'tensorboard.log'

dropout_rate = 0.5
optimizer = Adam
initial_learning_rate = 5e-4
output_weight = [0.34, 0.33, 0.33]
loss_function = dice_coefficient_loss
metrics_function = [dice_coefficient]
learning_rate_drop = 0.5
learning_rate_patience = 20
batch_size = 30
epochs = 10

def get_callback(model_file, log_file, learning_rate_drop, learning_rate_patience, tensorboard_log_file):
    callback = []
    callback.append(ModelCheckpoint(model_file, monitor = 'val_acc', verbose = 1, mode = 'max'))
    callback.append(CSVLogger(log_file, append = False))
    callback.append(ReduceLROnPlateau(factor = learning_rate_drop,
                                      patience = learning_rate_patience,
                                      verbose = 1))
    callback.append(TensorBoard(log_dir = tensorboard_log_file))
    return callback

def train(output_path = output_path,
          batch_size = batch_size,
          epochs = epochs,
          optimizer = optimizer,
          initial_learning_rate = initial_learning_rate,
          output_weight = output_weight,
          loss_function = loss_function,
          metrics_function = metrics_function,
          learning_rate_drop = learning_rate_drop,
          learning_rate_patience = learning_rate_patience):

    global train_patch_name, label_patch_name, check_point, log_file, model_file
    global tensorboard_log_file

    model = u_net3d_model(dropout_rate = dropout_rate,
                          optimizer = optimizer,
                          initial_learning_rate = initial_learning_rate,
                          output_weight = output_weight,
                          loss_function = loss_function,
                          metrics_function = metrics_function)

    train_patch = np.memmap(train_patch_name, mode = 'r', dtype = np.float32)
    train_patch = train_patch.reshape(-1, 4, 33, 33, 33)
    label_patch = np.memmap(label_patch_name, mode = 'r', dtype = np.int8,
                            shape = (train_patch.shape[0], 1))
    label_patch = to_categorical(label_patch, num_classes = 5)


    model.fit(x = train_patch,
              y = label_patch,
              batch_size = batch_size,
              epochs = epochs,
              callbacks = get_callback(model_file = check_point,
                                       log_file = log_file,
                                       learning_rate_drop = learning_rate_drop,
                                       learning_rate_patience = learning_rate_patience
                                       tensorboard_log_file = tensorboard_log_file))

    model.save(model_file)

if __name__ == '__main__':
    train()

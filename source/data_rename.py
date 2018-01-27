# -*- coding: utf-8 -*-
# python 2
"""
Brats2015 data의 파일명을 정렬하는 과정이다.
train data[Flair, T1, T1c, T2, OT] : HGG(220 samples), LGG(54 samples)
test data[Flair, T1, T1c, T2] : HGG_LGG(110 samples)
"""
from __future__ import print_function

import os
import shutil
import SimpleITK as sitk

data_path = input('Input your data directory : ')
data_path = os.path.abspath(data_path)

a = True
while a:
    try:
        HGG = os.listdir(data_path + '/HGG/')
        LGG = os.listdir(data_path + '/LGG/')
        TEST = os.listdir(data_path + '/HGG_LGG/')
        a = False
    except FileNotFoundError:
        data_path = input('Input data directory that contains HGG, LGG and HGG_LGG : ')

for i in range(len(HGG)):
    subfolder = os.listdir(data_path + '/HGG/' + HGG[i] + '/')
    for j in range(len(subfolder)):
        datafile = os.listdir(data_path + '/HGG/' + HGG[i] + '/' + subfolder[j] + '/')
        for k in datafile:
            if k.endswith('mha'):
                shutil.copy(data_path + '/HGG/' + HGG[i] + '/' + subfolder[j] + '/' + k, data_path)
                os.rename(data_path + '/' + k, data_path + '/' + 'HGG.' + str(i) + '.' + k)
                print(k, 'renamed to', 'HGG.' + str(i) + '.' + k)

for i in range(len(LGG)):
    subfolder = os.listdir(data_path + '/LGG/' + LGG[i] + '/')
    for j in range(len(subfolder)):
        datafile = os.listdir(data_path + '/LGG/' + LGG[i] + '/' + subfolder[j] + '/')
        for k in datafile:
            if k.endswith('mha'):
                shutil.copy(data_path + '/LGG/' + LGG[i] + '/' + subfolder[j] + '/' + k, data_path)
                os.rename(data_path + '/' + k, data_path + '/'  + 'LGG.' + str(i) + '.' + k)
                print(k, 'renamed to', 'LGG.' + str(i) + '.' + k)

for i in range(len(TEST)):
    subfolder = os.listdir(data_path + '/HGG_LGG/' + TEST[i] + '/')
    for j in range(len(subfolder)):
        datafile = os.listdir(data_path + '/HGG_LGG/' + TEST[i] + '/' + subfolder[j] + '/')
        for k in datafile:
            if k.endswith('mha'):
                shutil.copy(data_path + '/HGG_LGG/' + TEST[i] + '/' + subfolder[j] + '/' + k, data_path)
                os.rename(data_path + '/' + k, data_path + '/' + 'TEST.' + str(i) + '.' + k)
                print(k, 'renamed to', 'TEST.' + str(i) + '.' + k,)

shutil.rmtree(data_path + '/HGG')
shutil.rmtree(data_path + '/LGG')
shutil.rmtree(data_path + '/HGG_LGG')

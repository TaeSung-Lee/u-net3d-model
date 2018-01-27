# -*- coding: utf-8 -*-
# python 2

"""
'OT' modality data 파일에 대해서 파일의 확장자를 변경한다. (mha to nii)
pixel의 dtype를 int16(-32768 ~ 32768) 에서 uint8(0 ~ 255)로 변경한다.
"""
from __future__ import print_function

import os
import SimpleITK as sitk
import numpy as np

data_path = input('Input data path : ')
data_path = os.path.abspath(data_path)

os.mkdir(data_path + '/data/')

files = [f for f in os.listdir(data_path) if os.path.isfile(data_path + '/' + f)]

for filename in files:
    if 'OT' in filename:
        sitk_img = sitk.ReadImage(data_path + '/' + filename)
        img = sitk.GetArrayFromImage(sitk_img).astype('uint8')
        sitk_img = sitk.GetImageFromArray(img)
        nonExtenName = '.'.join(filename.split('.')[:-1])
        sitk.WriteImage(sitk_img, data_path + '/data/' + nonExtenName + '.nii')
        print(filename, 'TO', nonExtenName + '.nii')
        os.remove(data_path + '/' +  filename)

print(
"""
Flair, T1, T1c, T2 modality datas have to be processed by 3D Slicer's N4ITKBiasFieldCorrection tool
3D Slicer : http://download.slicer.org/

$ cd your data[Flair, T1, T1c, T2] directory
$ for n in *.mha; do ~/slicer/lib/Slicer-4.7/cli-modules/N4ITKBiasFieldCorrection "./$n" ./data/"${n%.mha}.nii"; done
""")

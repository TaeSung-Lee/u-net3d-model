from __future__ import print_function

import os
import SimpleITK as sitk
from time import localtime, strftime

data_path = './before_N4/'
data_list = os.listdir(data_path)
output_path = './data/'

print(strftime('%Y-%m-%d  %H:%M:%S', localtime()), ': MRI N4BiasFieldCorrection Started.')

total = len(data_list)

cnt = 1
for sample in data_list:
    if '.OT.' in sample:
        print('\r', sample, ':', cnt, '/', total, end="")
        img = sitk.ReadImage(data_path + sample)
        sitk.WriteImage(img, output_path + sample)
        cnt += 1

    else:
        img = sitk.ReadImage(data_path + sample)
        mask = sitk.OtsuThreshold(img, 0, 1, 200)
        img = sitk.Cast(img, sitk.sitkFloat32)
        print('\r', sample, ':', cnt, '/', total, end="")
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        output = corrector.Execute(img, mask)
        sitk.WriteImage(output, output_path + sample)
        cnt += 1

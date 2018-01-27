import os
import numpy as np
import SimpleITK as sitk
from keras.models import load_model
from source.pre_process import get_wdh_and_pad, get_data_list, get_time
from source.metrics import (dice_coefficient, dice_coefficient_loss)

DATATYPE = {'HGG':0, 'LGG':1, 'TEST':2}
SHAPE = (240, 240, 155)
PATCH = (33, 33, 33)
MODS = 4
SEGMENTATION = 5

test_samples = 110
batch_size = 30

output_path = input('Input output path : ')
data_path = input('Input data path : ')

try:
    os.mkdir(output_path + 'prediction')
except OSError:
    pass

model_dir = output_path + 'model/'
model_path = model_dir + 'u_net3d_model.h5'
predict_path = output_path + 'prediction/'

model = load_model(model_path, custom_objects = {'dice_coefficient' : dice_coefficient,
                                                 'dice_coefficient_loss' : dice_coefficient_loss})

def test_predict():
    test_data = np.memmap(output_path + 'TEST.dat', mode = 'r', dtype = np.float32,
                          shape = (test_samples, MODS, SHAPE[0], SHAPE[1], SHAPE[2]))
    data_list = get_data_list(data_path)[2]
    name_list = []
    for sample in range(test_samples):
        name = data_list[sample, 0]
        VSD_id = name.split('.')[-2]
        new_name = 'VSD.test_predict_{:03}.{}.mha'.format(sample + 1, VSD_id)
        name_list.append(new_name)

    for sample in range(test_samples):
        prediction_sample = test_data[sample]
        indices = []
        input_patches =  np.zeros(shape = (batch_size, MODS, PATCH[0], PATCH[1], PATCH[2]), dtype = np.float32)
        mri = np.zeros(shape = (SHAPE[0], SHAPE[1], SHAPE[2]), dtype = np.int16)
        cnt = 1
        for w in range(SHAPE[0]):
            for d in range(SHAPE[1]):
                for h in range(SHAPE[2]):
                    idx = (w, d, h)
                    indices.append(idx)
                    wdh, pad = get_wdh_and_pad(idx)
                    patch = test_data[sample, :, wdh[0]:wdh[1], wdh[2]:wdh[3], wdh[4]:wdh[5]]
                    if patch.shape != (4, 33, 33, 33):
                        patch = np.pad(patch, pad_width = pad, mode = 'constant', constant_values = 0)
                    input_patches[indices.index(idx)] = patch
                    if len(indices) == 30 or idx == (240, 240, 155):
                        prediction = model.predict(x = input_patches, batch_size = batch_size, verbose = 0)
                        for idx in indices:
                            mri[idx] = np.argmax(prediction[indices.index(idx)])
                        print(get_time() + ' : creating {}, {}%'.format(name_list[sample],
                                                                        np.float32(cnt) / np.float32(297600) * 100))
                        cnt += 1
                        indices = []
                        input_patches[:, :, :, :, :] = 0
        sitk_img = sitk.GetImageFromArray(mri.transpose())
        sitk.WriteImage(sitk_img, predict_path + name_list[sample])

def manual_predict():
    sample = input('''
Input sample TYPE and NUMBER
    HGG : 0 ~ 219
    LGG : 0 ~ 54
    TEST : 0 ~ 109
ex) 'HGG,10'
''')
    sample_type, sample_number = sample.split(',')
    sample_number = int(sample_number)
    data_list = get_data_list(data_path)[DATATYPE[sample_type]]
    name_list = [data_list[sample_number, mod] for mod in range(4)]

    sample_data = get_data(sample_type, sample_number, name_list) # SHAPE = (4, 240, 240, 155)
    name = name_list[0]
    VSD_id = name.split('.')[-2]
    new_name = 'VSD.test_predict_{:03}.{}.mha'.format(sample_number + 1, VSD_id)

    indices = []
    input_patches = np.zeros(shape = (batch_size, MODS, PATCH[0], PATCH[1], PATCH[2]), dtype = np.float32)
    prediction_image = np.zeros(shape = (SHAPE[0], SHAPE[1], SHAPE[2]), dtype = np.int16)

    print(get_time() + ' : creating {}'.format(new_name))
    for w in range(SHAPE[0]):
        for d in range(SHAPE[1]):
            for h in range(SHAPE[2]):
                idx = (w, d, h)
                indices.append(idx)
                wdh, pad = get_wdh_and_pad(idx)
                patch = sample_data[:, wdh[0]:wdh[1], wdh[2]:wdh[3], wdh[4]:wdh[5]]
                if patch.shape != (4, 33, 33, 33):
                    patch = np.pad(patch, pad_width = pad, mode = 'constant', constant_values = 0)
                input_patches[indices.index(idx)] = patch
                if len(indices) == 30 or idx == (240, 240, 155):
                    prediction = model.predict(x = input_patches, batch_size = batch_size, verbose = 0)
                    for idx in indices:
                        prediction_image[idx] = np.argmax(prediction[indices.index(idx)])
                    print(get_time() + ' : creating {}, {}%'.format(name_list[sample],
                                                                    np.float32(cnt) / np.float32(297600) * 100))
                    cnt += 1
                    indices = []
                    input_patches[:, :, :, :, :] = 0
    sitk_img = sitk.GetImageFromArray(prediction_image.transpose())
    sitk.WriteImage(sitk_img, predict_path + new_name)


def get_data(data_type, data_number):
    fp = np.memmap(predict_path + '{}.{:03}.dat'.format(data_type, data_number), mode = 'w+',
                   dtype = np.flaot32, shape = (MODS, SHAPE[0], SHAPE[1], SHAPE[2]))
    with open(output_path + 'Flair_irs.pkl', 'r') as f1:
        flair_irs = pickle.load(f1)
    with open(output_path + 'T1_irs.pkl', 'r') as f2:
        t1_irs = pickle.load(f2)
    with open(output_path + 'T1c_irs.pkl', 'r') as f3:
        t1c_irs = pickle.load(f3)
    with open(output_path + 'T2_irs.pkl', 'r') as f4:
        t2_irs = pickle.load(f4)
    for mod in range(MODS):
        img = get_image(name_list[mod])
        if mod == 0:
            img[img > 0] = flair_irs.transform(img[img > 0], surpress_mapping_check = True)
        elif mod == 1:
            img[img > 0] = t1_irs.transform(img[img > 0], surpress_mapping_check = True)
        elif mod == 2:
            img[img > 0] = t1c_irs.transform(img[img > 0], surpress_mapping_check = True)
        elif mod == 4:
            img[img > 0] = t2_irs.transform(img[img > 0], surpress_mapping_check = True)
        img[img == 0 ] = 0
        minpix = np.min(img)
        if minpix < 0:
            img[img != 0] -= minpix
        fp[mod] = img
        if mod == MOD['MR_T1']:
            img_e = histogram_equalizing(img.astype(np.uint16))
            fp[mod] = img_e.astype(np.float32)
        else:
            fp[mod] = img
    means, stds = np.load(output_path + data_type + '_MeanAndStd.npy')
    for mod in range(4):
        fp[:, mod, :, :, :] -= means[mod]
        fp[:, mod, :, :, :] /= stds[mod]
    return fp

def get_image(img_name):
    path = data_path
    sitk_img = sitk.ReadImage(path + img_name)
    img = sitk.GetArrayFromImage(sitk_img).transpose()
    minpix = np.min(img)
    if minpix < 0:
        img[img != 0] -= minpix
    return img



if __name__ == '__main__':
    prediction_mod = int(input('Prediction MOD (0 : test prediction, 1 : manual prediction)'))
    if not prediction_mod:
        test_predict()
    else:
        manual_predict()

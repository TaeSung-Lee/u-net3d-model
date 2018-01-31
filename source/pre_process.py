# -*- coding: utf-8 -*-
# python 2
"""
Brats2015 data를 U_net3d의 사용에 적합한 데이터로 가공하는 과정.
다음의 세 종류의 전처리 과정을 포함한다.

1. IntensityRangeStandardization(funcs: get_trained_irs, get_trained_data)
MRI의 특성상 촬영하는 환자와 장치가 동일하더라도, 얻어지는 MRI의 intensity range의 차이가 심하다.
하지만 CNN model에 input되는 데이터의 intensity range가 일정해야 정확한 예측값을 기대할 수 있다.
따라서 medpy package의 IntensityRangeStandardization 툴을 사용하여 data의 intensity range를 보정해준다.
Brats2015 data는 각 modality(flair, t1, t1c, t2)별로 표준강도공간(common intensity range)의 irs model을 학습한다.
다음에 training, test MRI data의 해당하는 modality irs model을 이용하여 intensity range를 보정해준다.

2. histogram equalizing (funcs: histogram_equalizing, data_histogram_equalizing)
컴퓨터 이미지는 일정한 범위의 intensity로(일반적인 흑백 이미지는 0 ~ 255의 숫자로 표현) 표현된다.
이미지의 히스토그램이 특정 범위에만 집중될 경우 해당 이미지의 contrast가 저하되고,
이미지를 인식하는 neural network의 성능 저하가 예상된다.
histogram equalizing은 특정 영역에 몰려있는 히스토그램을 이미지의 전 강도 공간에 고루 분포할 수 있도록 보정해준다.

3. gauss Normalization (funcs: get_mean_and_std, gauss_normalization)
이미지 데이터를 정규화 함으로써 CNN의 성능 향상을 기대할 수 있다.
따라서 데이터의 type(HGG, LGG, TEST)과 modality(flair, t1, t1c t2)별로 평균과 표준편차를 계산한다.
이미지 데이터에서 평균을 빼주고, 표준편차를 나눠줌으로써, 데이터가 zero mean, unit standard deviation을 갖도록 정규화해준다.

label데이터('OT')의 segmentation(0:everything else, 1:necrosis, 2:edema, 3:enhancing tumor, 4:non-enhancing tumor) 별로
index(patch를 생성하는 중심 pixel의 3차원 좌표)를 생성한다. (func: get_label_index)

전처리가 끝난 데이터는 label index를 중심으로 33x33x33 크기의 patch를
4 channels(flari, t1, t1c, t2가 해당하며, u_net3d model 에 동시에 input된다)로 생성한다.

CNN을 학습시에는 많은 양의 이미지 데이터를 요구하게 된다. 이미지 데이터를 증폭하기 위해서
다양한 조작(rotation, flip, zoom etc..)이 사용된다. 여기서는 90도 회전을 통해 데이터를 4배로 증폭하여 사용한다. (func: get_patch)

최종적으로 U-net3d model 학습에 사용되는 HGG, LGG의 training patch data, label data와 test patch data를 생성한다.
"""
from __future__ import print_function

import os
import pickle
from time import localtime, strftime

import numpy as np
import SimpleITK as sitk
from medpy.filter import IntensityRangeStandardization
from medpy.io import save

MOD = {'MR_Flair':0, 'MR_T1':1, 'MR_T1c':2, 'MR_T2':3, 'OT':4} # 각 modality에 해당하는 number
MODS = {'HGG':5, 'LGG':5, 'TEST':4} # HGG와 LGG의 데이터는 OT데이터를 포함하기 때문에 5개의 데이터 modality가 존재한다.
DATATYPE = {'HGG':0, 'LGG':1, 'TEST':2}
SHAPE = [240, 240, 155] # MRI 데이터의 크기를 정의한다.
SEGMENTATION = 5 # 0:everything else, 1:necrosis, 2:edema, 3:non-enhancing tumor, 4:enhancing tumor
INDEXSIZE = [100, 100, 100, 100, 100]
# label index들의 데이터 크기를 정의한다. INDEXSIZE에 따라 데이터의 구성 비율이 바뀌게 되면
# 학습되는 model의 결과에도 영향을 미치게 된다.
PATCH = [33, 33, 33] # model에 input되는 이미지의 크기를 정의한다.
ROTATE = 4 # 이미지 회전의 횟수.



# 소스코드 실행 시 시간을 기록하여, 경과시간을 표시한다.
def get_time():
    return strftime('%Y-%m-%d  %H:%M:%S', localtime())

'''
path에 존재하는 MRI데이터 이름을 타입(HGG, LGG, TEST)과 모드(flari, t1, t1c, t2)에 따라서 정렬한 후 리턴한다.
'''
def get_data_list(path):
    name_list = os.listdir(path)
    HGG = {}
    LGG = {}
    TEST = {}
    for filename in name_list:
        temp = filename.split('.')
        typ = temp[0]
        cnt = int(temp[1])
        mod = int(MOD[temp[-3]])
        if typ == "HGG":
            HGG[cnt, mod] = filename
        elif typ == 'LGG':
            LGG[cnt, mod] = filename
        else: #TEST(HGG_LGG)
            TEST[cnt, mod] = filename
    return np.array([HGG, LGG, TEST])

'''
Brats 2015데이터는 mha확정자를 갖는 파일이다. 이를 가공과 학습에 용이한 형태인 numpy array로 가져오기 위해서
SimpleITK module의 ReadImage와 GetArrayFromImage 함수들을 이용한다.
대부분의 MRI데이터의 intensity range는 0을 포함하는 양수로 표현된다. 하지만 일부 데이터에는 음수로 표현되는 경우가 있다.
이를 보정해주기 위해서 해당 MRI데이터의 최소값을 0을 제외한 전체 pixel값에서 빼줌으로써, 이미지의 intensity range를 양의 영역으로 가져온다.
'''
def get_image(img_name, data_path):
    path = data_path
    sitk_img = sitk.ReadImage(path + img_name)
    img = sitk.GetArrayFromImage(sitk_img).transpose()
    minpix = np.min(img)
    if minpix < 0:
        img[img != 0] -= minpix
    return img
'''
nii, mha 확장자의 MRI 파일을 읽어드려, numpy memmap 형태로 저장한다.
ex)
    memmap.shape : (total_patients, number of modality, 240, 240, 155)
'''
def get_orig_data(data_list, dataClass, output_path, data_path):
    mods = MODS[dataClass]
    total = len(data_list[DATATYPE[dataClass]]) // mods
    if mods == 5:
        mods = 4
    fp = np.memmap(output_path + dataClass + '_orig.dat', dtype = np.float32, mode = 'w+',
                   shape = (total, mods, SHAPE[0], SHAPE[1], SHAPE[2]))
    for sample in range(total):
        for mod in range(mods):
            img = get_image(data_list[DATATYPE[dataClass]][sample, mod], data_path)
            fp[sample, mod] = img
        print('\r', get_time() + ': {} getting numpy array image {} / {}'.format(dataClass, sample + 1, total), end = '')

'''
irs(표준강도공간 model)을 학습하는 함수이다.
data는 각 modality별로 별도의 irs model에 학습된다.

parameters : cutoff : (float, float)
                 Lower and upper cut-off percentiles to exclude outliers.
             landmarks : sequence of floats
                 List of percentiles serving as model landmarks, must lie between the cutoffp values.
reference : http://loli.github.io/medpy/generated/medpy.filter.IntensityRangeStandardization.IntensityRangeStandardization.html
'''
def get_trained_irs(data_list, output_path, cutoffp = (1, 20),
                    landmarkp = [2,3,4,5,6, 8,10,12,14, 15,16,17,18,19]): # Default : cutoffp = (1, 99), landmarkp = [10, 20, 30, 40, 50, 60, 70, 90]
    flair_irs = IntensityRangeStandardization(cutoffp = cutoffp, landmarkp = landmarkp)
    t1_irs    = IntensityRangeStandardization(cutoffp = cutoffp, landmarkp = landmarkp)
    t1c_irs   = IntensityRangeStandardization(cutoffp = cutoffp, landmarkp = landmarkp)
    t2_irs    = IntensityRangeStandardization(cutoffp = cutoffp, landmarkp = landmarkp)

    for dataClass in ['HGG', 'LGG', 'TEST']:
        mods = MODS[dataClass]
        total = len(data_list[DATATYPE[dataClass]]) // mods
        if mods == 5: # OT data는 별도의 전처리과정을 필요로 하지 않으므로, 따리 관리하도록 한다.
            mods = 4
        fp = np.memmap(output_path + dataClass + '_orig.dat', dtype = np.float32, mode = 'r',
                       shape = (total, mods, SHAPE[0], SHAPE[1], SHAPE[2]))
        print('\r', get_time() + ': training irs with {} images'.format(dataClass))
        # 이미 사전에 학습된 표준강도공간이 존재하면 이를 불러와서 계속해서 학습한다.
        for mod in range(mods):
            images = fp[:, mod, :, :, :]
            if mod == MOD['MR_Flair']:
                flair_irs = flair_irs.train([images[images > 0]])
            elif mod == MOD['MR_T1']:
                t1_irs = t1_irs.train([images[images > 0]])
            elif mod == MOD['MR_T1c']:
                t1c_irs = t1c_irs.train([images[images > 0]])
            elif mod == MOD['MR_T2']:
                t2_irs = t2_irs.train([images[images > 0]])
    with open(output_path + 'Flair_irs.pkl', 'wb') as f1:
        pickle.dump(flair_irs, f1)
    with open(output_path + 'T1_irs.pkl', 'wb') as f2:
        pickle.dump(t1_irs, f2)
    with open(output_path + 'T1c_irs.pkl', 'wb') as f3:
        pickle.dump(t1c_irs, f3)
    with open(output_path + 'T2_irs.pkl', 'wb') as f4:
        pickle.dump(t2_irs, f4)

'''
MRI 데이터 중 label data인 OT modality는 별도의 전처리 과정을 필요로 하지 않는다.
따라서 데이터를 따로 관리하도록 한다.
HGG와 LGG 샘플 타입의 경우에만 OT 데이터가 존재한다.
'''
def get_label_data(dataClass, dataList, output_path, data_path):
    mods = MODS[dataClass]
    total = len(dataList[DATATYPE[dataClass]]) // mods
    fp = np.memmap(output_path + dataClass + '_label.dat', mode = 'w+', dtype = np.int8,
                   shape = (total, SHAPE[0], SHAPE[1], SHAPE[2]))
    for sample in range(total):
        label = get_image(dataList[DATATYPE[dataClass]][sample, mods - 1], data_path)
        fp[sample] = label.astype(np.uint8)
        print('\r', get_time() + ': {} getting label data {} / {}'.format(dataClass, sample + 1, total), end = '')

'''
MRI 데이터 샘플별로 'OT'데이터의 label index의 좌표값들은 구한다.
0:everything else, 1:necrosis, 2:edema, 3:non-enhancing tumor, 4:enhancing tumor
입력되는 indexsize는 list타입이다.
'''
def get_label_index(dataClass, data_list, indexSize, output_path):
    mods = MODS[dataClass]
    total = len(data_list[DATATYPE[dataClass]]) // mods
    label_data = np.memmap(output_path + dataClass + '_label.dat', mode = 'r', dtype= np.int8,
                   shape = (total, SHAPE[0], SHAPE[1], SHAPE[2]))
    seg = SEGMENTATION
    label = {}
    for sample in range(total):
        for segment in range(seg):
            # img = get_image(data_list[DATATYPE[dataClass]][sample, mods - 1])
            img = label_data[sample]
            label_idx = np.argwhere(img == segment).astype(np.uint8)
            label_idx = np.random.permutation(label_idx)
            label[sample, segment] = label_idx[:indexSize[segment]].astype(np.uint8)
    with open(output_path + dataClass +'_labelindex.pkl', 'wb') as f:
        pickle.dump(label, f)
        print('\r', get_time() + ': {} getting label indice {} / {}'.format(dataClass, sample + 1, total), end = '')

'''
MRI 데이터를 타입별로 intensity range를 standard intensity range로 대체하고,
T1 modality에 대해서 histogram equalizing을 실행한다.
label data인 OT data는 별도의 전처리 과정을 필요로 하지 않으므로, 생략을 한다.
'''
def get_data(dataClass, dataList, output_path, data_path):
    mods = MODS[dataClass]
    total = len(dataList[DATATYPE[dataClass]]) // mods
    if mods == 5:
        mods = 4
    fp = np.memmap(output_path + dataClass + '.dat', mode = 'w+', dtype = np.float32,
                   shape = (total, mods, SHAPE[0], SHAPE[1], SHAPE[2]))
    # with open(output_path + 'HGGModel.pkl', 'r') as f:
    #     irs = pickle.load(f)
    with open(output_path + 'Flair_irs.pkl', 'rb') as f1:
        flair_irs = pickle.load(f1)
    with open(output_path + 'T1_irs.pkl', 'rb') as f2:
        t1_irs = pickle.load(f2)
    with open(output_path + 'T1c_irs.pkl', 'rb') as f3:
        t1c_irs = pickle.load(f3)
    with open(output_path + 'T2_irs.pkl', 'rb') as f4:
        t2_irs = pickle.load(f4)
    for sample in range(total):
        for mod in range(mods):
            img = get_image(dataList[DATATYPE[dataClass]][sample, mod], data_path)
            if mod == MOD['MR_Flair']:
                img[img > 0] = flair_irs.transform(img[img > 0], surpress_mapping_check = True)
            elif mod == MOD['MR_T1']:
                img[img > 0] = t1_irs.transform(img[img > 0], surpress_mapping_check = True)
            elif mod == MOD['MR_T1c']:
                img[img > 0] = t1c_irs.transform(img[img > 0], surpress_mapping_check = True)
            elif mod == MOD['MR_T2']:
                img[img > 0] = t2_irs.transform(img[img > 0], surpress_mapping_check = True)
            minpix = np.min(img)
            if minpix < 0:
                img[img != 0] -= minpix
            if mod == MOD['MR_T1']:
                img_e = histogram_equalizing(img.astype(np.uint16))
                fp[sample, mod] = img_e.astype(np.float32)
            else:
                fp[sample, mod] = img
        print('\r', get_time() + ': {} geting processed image {} / {}'.format(dataClass, sample + 1, total), end = '')

'''
MRI 데이터의 환자 타입과 이미지 modality별로 표준편차와 평균값을 계산한다.
'''
def get_mean_and_std(dataClass, dataList, output_path):
    mods = MODS[dataClass]
    total = len(dataList[DATATYPE[dataClass]]) // mods
    if mods == 5:
        mods = 4
    fp = np.memmap(output_path + dataClass + '.dat', mode = 'r', dtype = np.float32,
                   shape = (total, mods, SHAPE[0], SHAPE[1], SHAPE[2]))
    stds = np.zeros(mods, dtype = np.float32)
    for mod in range(mods):
        stds[mod] = np.std(fp[:, mod, :, :, :])
    means = np.mean(fp, axis=(0,2,3,4))
    mean_and_std = np.array([means, stds]).astype(np.float32)
    np.save(output_path + dataClass + '_MeanAndStd.npy', mean_and_std)
    print(get_time() + ': %s got mean and std' %(dataClass))


'''
입력된 이미지의 히스토크램을 equalizing한다.

reference : http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html#histogram-equalization
'''
def histogram_equalizing(img):
    maxpix = np.max(img)
    # bins : hist에 해당하는 pixel intensity, intensity range가 0부터 시작하므로, bins의 최대값은 maxpix+1이 된다
    hist, bins = np.histogram(img.flatten(), maxpix+1, [0, maxpix+1])
    cdf = hist.cumsum() # histogram의 누적분배함수 (cdf : Cumulative Distribution Function)
    # cdf_normalized = cdf * hist.max() / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * maxpix / (cdf_m.max() - cdf_m.min())
    cdf_e = np.ma.filled(cdf_m, 0).astype(np.uint16)
    img_e = cdf_e[img]

    # hist_e, bins_e = np.histogram(img_e.flatten(), maxpix+1, [0,maxpix+1])
    # cdf_normalized = cdf_e * hist_e.max() / cdf_e.max()
    return img_e

'''
idx 좌표를 기준으로 (33, 33, 33) size의 patch를 생성하기위한
w : width, d : depth, h : height 의 좌표를 반환한다.
idx 가 이미지의 코너 부근에 존재할 경우 생성된 patch의 size가 (33, 33, 33) 보다
작을 수 있다. 이 경우에는 누락된 사이즈 만큼 0의 값들로 padding 한다.
'''
def get_wdh_and_pad(idx):
    w_d_h = [idx[0] - PATCH[0]/2, idx[0] + PATCH[0]/2 + 1,
             idx[1] - PATCH[1]/2, idx[1] + PATCH[1]/2 + 1,
             idx[2] - PATCH[2]/2, idx[2] + PATCH[2]/2 + 1]
    w_d_h2= [max(0, w_d_h[0]), min(240, w_d_h[1]), max(0, w_d_h[2]), min(240, w_d_h[3]),
             max(0, w_d_h[4]), min(155, w_d_h[5])]
    pad = ((0, 0),
           (abs(w_d_h2[0] - w_d_h[0]), abs(w_d_h2[1] - w_d_h[1])),
           (abs(w_d_h2[2] - w_d_h[2]), abs(w_d_h2[3] - w_d_h[3])),
           (abs(w_d_h2[4] - w_d_h[4]), abs(w_d_h2[5] - w_d_h[5])))
    return w_d_h2, pad

'''
u_net3d model 학습에 사용되는 training patch와 target patch 를 생성한다.
생성된 pacth가 3차원 이미지 이므로 rotate는 각 축(x, y, z) 별로 1회씩 실시하도록 한다.
'''
def get_patch(dataClass, data_list, output_path):
    mods = MODS[dataClass]
    total = len(data_list[DATATYPE[dataClass]]) // mods
    if mods == 5:
        mods = 4
    fp = np.memmap(output_path + dataClass + '.dat', mode = 'r', dtype = np.float32,
                   shape = (total, mods, SHAPE[0], SHAPE[1], SHAPE[2]))
    with open(output_path + dataClass + '_labelindex.pkl', 'rb') as f:
        label = pickle.load(f)
    sample_size = np.sum([index.shape[0] for _, index in label.items()])
    fp_train = np.memmap(output_path + dataClass + '_train.pat', mode = 'w+', dtype = np.float32,
                         shape = (sample_size * ROTATE, mods, PATCH[0], PATCH[1], PATCH[2]))
    fp_label = np.memmap(output_path + dataClass + '_label.pat', mode = 'w+', dtype = np.int8,
                         shape = (fp_train.shape[0]))
    segment = SEGMENTATION
    cnt = 0
    for sample in range(total):
        for seg in range(segment):
            idxs = label[sample, seg]
            for idx in idxs:
                w_d_h, pad = get_wdh_and_pad(idx)
                patch = fp[sample, :, w_d_h[0]:w_d_h[1], w_d_h[2]:w_d_h[3], w_d_h[4]:w_d_h[5]]
                if patch.shape != (4, 33, 33, 33):
                    patch = np.pad(patch, pad_width = pad, mode = 'constant', constant_values = 0)
                for rot in range(ROTATE):
                    if rot == 0:
                        fp_train[cnt] = patch
                        fp_label[cnt] = seg
                        cnt += 1
                    elif rot == 1:
                        fp_train[cnt] = np.rot90(patch, 1, (1,2))
                        fp_label[cnt] = seg
                        cnt += 1
                    elif rot == 2:
                        fp_train[cnt] = np.rot90(patch, 1, (1,3))
                        fp_label[cnt] = seg
                        cnt += 1
                    elif rot == 3:
                        fp_train[cnt] = np.rot90(patch, 1, (2,3))
                        fp_label[cnt] = seg
                        cnt += 1
        print('\r', get_time() + ': {} getting a patch {} / {}'.format(dataClass, sample + 1, total), end = '')

    state = np.random.get_state()
    print(get_time() + ': %s patch data is being shuffled' %(dataClass))
    np.random.shuffle(fp_train)
    np.random.set_state(state)
    print(get_time() + ': %s label data is being shuffled' %(dataClass))
    np.random.shuffle(fp_label)


'''
생성된 data를 gauss Normalization을 한다.
'''
def gauss_norm(dataClass, output_path):
    mods = MODS[dataClass]
    if mods == 5:
        mods = 4
    if dataClass in ['HGG', 'LGG']:
        fp = np.memmap(output_path + dataClass + '_train.pat', mode = 'r+', dtype = np.float32)
        fp = fp.reshape(-1, mods, PATCH[0], PATCH[1], PATCH[2])
    else:
        fp = np.memmap(output_path + dataClass + '.dat', mode = 'r+', dtype = np.float32)
        fp = fp.reshape(-1, mods, SHAPE[0], SHAPE[1], SHAPE[2])
    means, stds = np.load(output_path + dataClass + '_MeanAndStd.npy')
    for mod in range(mods):
        fp[:, mod, :, :, :] -= means[mod]
        fp[:, mod, :, :, :] /= stds[mod]
        print('\r', get_time() + ': {}`s {} training data is being normalized'.format(dataClass, mod), end = '')

def main_process():
    data_path = input('input data path : ')
    output_path = input('input output path : ')
    data_list = get_data_list(data_path)
    get_orig_data(data_list, 'HGG', output_path, data_path)
    get_orig_data(data_list, 'LGG', output_path, data_path)
    get_orig_data(data_list, 'TEST', output_path, data_path)
    get_trained_irs(data_list, output_path)
    get_label_data('HGG', data_list, output_path, data_path)
    get_label_index('HGG', data_list, INDEXSIZE, output_path)
    get_label_data('LGG', data_list, output_path, data_path)
    get_label_index('LGG', data_list, INDEXSIZE, output_path)
    get_data('HGG', data_list, output_path, data_path)
    get_data('LGG', data_list, output_path, data_path)
    get_data('TEST', data_list, output_path, data_path)
    get_mean_and_std('HGG', data_list, output_path)
    get_mean_and_std('LGG', data_list, output_path)
    get_mean_and_std('TEST', data_list, output_path)
    get_patch('HGG', data_list, output_path)
    get_patch('LGG', data_list, output_path)
    gauss_norm('HGG', output_path)
    gauss_norm('LGG', output_path)
    gauss_norm('TEST', output_path)
    print('-'*20, 'all programs done', '-'*20)
if __name__ == '__main__':
    main_process()

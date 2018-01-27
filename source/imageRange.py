from pre_process import *

data_list = get_data_list(data_path)

rangeList = {}
HGG = {}
LGG = {}
TEST = {}

for mod in range(4):
    for sample in range(220):
        img = get_image(data_list[0][sample, mod])
        minpix = np.min(img)
        maxpix = np.max(img)
        HGG[sample, mod] = (minpix, maxpix)

for mod in range(4):
    for sample in range(54):
        img = get_image(data_list[1][sample, mod])
        minpix = np.min(img)
        maxpix = np.max(img)
        LGG[sample, mod] = (minpix, maxpix)

for mod in range(4):
    for sample in range(110):
        img = get_image(data_list[2][sample, mod])
        minpix = np.min(img)
        maxpix = np.max(img)
        TEST[sample, mod] = (minpix, maxpix)

rangeList['HGG'] = HGG
rangeList['LGG'] = LGG
rangeList['TEST'] = TEST

np.save(output_path + 'imageRange.npy', rangeList)

# Brain tumor segementaion with U-net 3D neural network models

##Requirements
* python2.7
* SimpleITK
* [Medpy](http://loli.github.io/medpy/)
* keras
* h5py
* [3DSlicer](https://www.slicer.org/)
##Dataset
Using [Brats 2015 dataset](https://www.smir.ch/BRATS/Start2015)

##Usage
As follows make diretories.
Unzip brats dataset in data directory.

>U-net3D
>>data
>>>HGG
>>>LGG
>>>HGG_LGG
>>output
>>source

###Pre-process
1. Run data_rename.py in U-net3D directory.
>This will rename brats dataset.

2. Run mha_to_nii.py
>This will change MRI ground truth file extension(.mha to .nii) and move file to /data/data directory

3. Using 3D slicer for doing N4ITKBiasFieldCorrection.
>N4N4ITKBiasFieldCorrection remove image gradation in your MRI dataset.

```bash
cd /U-net3d/data
for n in *.mha; do ~/slicer/lib/Slicer-4.7/cli-modules/N4ITKBiasFieldCorrection "./$n" ./data/"${n%.mha}.nii"; done
```
>3D slicer's file root maybe diffenent.

4. Run pre_process.py
>This source contains three pre-processes
>>Intensity Range Standardization
>>Histogram equalizing
>>Gauss Normalization

###Training
Run train.py

###Prediction
Run predict.py

##reference
* 윤지석, 석홍일 (2016). 딥러닝 기반의 멀티-모달 MRI 영상에서의 뇌종양 영역 분할. 한국정보과학회 학술발표논문집, 1680-1682
* Kayalibay, B., Jensen, G., & van der Smagt, P. (2017). CNN-based segmentation of medical imaging data. arXiv preprint arXiv:1701.03056.

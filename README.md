# Car_parts_segmentation_U-net
This repository is a project to create a segmentation model for segmenting car parts.

The architecture of the model will be U-net.
This architecture is easier to implement and requires less CPU/RAM than other architectures like FCN, ResNet, SegNet.
# How to use this repository
## Requirements
Before you start, you need to install the python libraries:
```
pip install -r ./0requirements.txt
```


## Dataset
After requirements, you need to **download repository** of DSMLR ( https://github.com/dsmlr/Car-Parts-Segmentation ). This will be the basis for the dataset.


For some reason, COCO annotation "showanns" was not working correctly on my PC.
See the photo below.

![image](.\img\image.png)

There were different results at each launch.
So I wrote a script that parses annotations and creates masks.

**Run the file 1makedataset.py**
The script will parse COCO annotations and create folders with the dataset.


As the dataset was created, I combined some classes, because 18 classes for 400 photos (training set) this was unnecessary.
 
The result of the work 1makedataset.py there will be a dataset folder with the train and val folders inside.
The dataset is ready for use.

## Augmentation
Initially, when training the model, it was planned to use Keras Image Data Generator to achieve better results. However, the RAM of my PC was not enough to use it with the model I modeled. Using a simplier model greatly worsens the results. Therefore, it was decided to use augmentation using a script.


When **2augmentation.py** finishes work the test folder will double in size with randomly changed images and masks. This augmentation increased the segmentation quality accuracy from 72 to 77.

## Train model
The model is created and trained in **3trainmodel.ipynb**

It displays information about the dataset and model.

The best weights of the model will also be saved in a folder for future reuse.

## How the model works
The notebook **4checkmodel.ipynb** checks the results of the model. 

Weights are loaded from the **model.h5** file, which was saved after the trainmodel notebook was running.
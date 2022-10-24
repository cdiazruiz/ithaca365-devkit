# Ithaca365 devkit
Welcome to the devkit of the [Ithaca365](https://ithaca365.mae.cornell.edu/) dataset.
![](https://ithaca365.mae.cornell.edu/files/2022/03/image7.jpg)

## Overview
- [Changelog](#changelog)
- [Devkit setup](#devkit-setup)
- [Ithaca365](#ithaca365)
  - [Ithaca365 setup](#Ithaca365-setup)
  - [Getting started with Ithaca365](#getting-started-with-ithaca365)
- [Known issues](#known-issues)
- [Citation](#citation)

## Devkit setup
We use a common devkit for Ithaca365.
The devkit is tested for Python 3.6, Python 3.7 and Python 3.8.

Our devkit is available and can be installed via [pip](https://pip.pypa.io/en/stable/installing/) :
```
pip install git+https://github.com/cxy1997/ithaca365-devkit.git
```

## Ithaca365

### Ithaca365 setup
To download Ithaca365 you need to go to the [Download page](https://ithaca365.mae.cornell.edu/).. 
For the devkit to work you will need to download *all* archives.
Please unpack the archives to the `/data/sets/ithaca365` folder \*without\* overwriting folders that occur in multiple archives.
Eventually you should have the following folder structure:
```
/data/sets/ithaca365
    samples	-	Sensor data for keyframes.
    sweeps	-	Sensor data for intermediate frames.
    v1.0-*	-	JSON tables that include all the meta data and annotations. Each split (trainval, test, mini) is provided in a separate folder.
```
If you want to use another folder, specify the `dataroot` parameter of the Ithaca365 class (see tutorial).

### Getting started with Ithaca365
Please follow these steps to make yourself familiar with the ithaca365 dataset:
- Read the [dataset description](https://ithaca365.mae.cornell.edu/).
- [Download](https://ithaca365.mae.cornell.edu/) the dataset. 
- Get the [ithaca365-devkit code]().
- Read the [online tutorial]() or run it yourself using:
```
jupyter notebook $HOME/ithaca365-devkit/tutorials/ithaca365_tutorial.ipynb
```
- Read the [ithaca365 paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Diaz-Ruiz_Ithaca365_Dataset_and_Driving_Perception_Under_Repeated_and_Challenging_Weather_CVPR_2022_paper.pdf) for a detailed analysis of the dataset.
- See the [database schema]().
- See the [FAQs]().

## Citation
Please use the following citation when referencing [Ithaca365]:
```
@InProceedings{Diaz-Ruiz_2022_CVPR,
    author    = {Diaz-Ruiz, Carlos A. and Xia, Youya and You, Yurong and Nino, Jose and Chen, Junan and Monica, Josephine and Chen, Xiangyu and Luo, Katie and Wang, Yan and Emond, Marc and Chao, Wei-Lun and Hariharan, Bharath and Weinberger, Kilian Q. and Campbell, Mark},
    title     = {Ithaca365: Dataset and Driving Perception Under Repeated and Challenging Weather Conditions},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {21383-21392}
}
```

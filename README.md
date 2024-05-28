![header](https://capsule-render.vercel.app/api?type=venom&text=Thymoma%20Segmentation&fontAlign=50&fontAlignY=50&fontSize=40&color=0:FBC2EB,100:A6C1EE&fontColor=white)


## Introduction
This is an implementation of the following paper.
> [Identification and risk classification of thymic epithelial tumors using 3D computed tomography images and deep learning models](https://www.sciencedirect.com/science/article/pii/S1746809424005317)
 Biomedical Signal Processing and Control (BSPC2024)

## Abstract
Thymic epithelial tumor (TET) is the most common neoplasm of the anterior mediastinum, accounting for approximately 47 % of all anterior mediastinal tumors. Early identification and risk stratification of patients with TET are beneficial for efficient intervention, but identifying subjects at high risk is challenging and time-consuming. This study proposes a deep-learning (DL) framework that combines the automatic segmentation of lung lesions with the classification of high-risk cases of TET. The model was trained, validated, and tested using computed tomography images of 125 patients with TET. Our method comprises two steps: 1) automatic segmentation of TET and comparison of the performance of 3D U-Net, 3D Res U-Net, 3D Dense U-Net, 3D Wide U-Net, and 3D U-Net++ models; 2) identification of high-risk TET cases using pretrained DL models (3D ResNet50, 3D SE ResNext50, 3D DenseNet121, and 3D VGG19) with transfer learning. The hyperparameters for segmentation and classification were determined using Bayesian optimization. Among the abovementioned 3D U-Net-based methods, 3D U-Net++ showed the best segmentation performance on the test set. Among the four classification models, ResNet50 performed the best on the test set. We also used a gradient-weighted class activation map to visualize the classification of risk groups in convolutional neural networks (CNNs), emphasizing sensitive features in the learning process. This study is the first to integrate automated segmentation of TETs with the identification of high-risk cases. The results demonstrate that CNN DL techniques can be used for the diagnosis and classification of risk groups in patients with TET.


## Copyright in machine learning and fluid mechanics Lab.

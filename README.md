![header](https://capsule-render.vercel.app/api?type=slice&text=Thymoma%20Segmentation&fontAlign=70&fontSize=40)

## Copyright in machine learning and fluid mechanics Lab.
## Introduction
This project is an implementation of [this paper](https://www.sciencedirect.com/science/article/pii/S1746809424005317).

Segmentation of the thymic epithelial tumor (TET) region containing the thymoma is essential for the identification of TET patients. However, this process is time consuming and difficult for radiologists. Therefore, in this paper, we intend to automatically segment regions of interest by applying deep learning techniques. Since the secured TET dataset is not sufficient, 3D U-Net-based models that show excellent performance even in small datasets were applied and compared.In addition, K-fold cross validation and Bayesian optimization were used. As a result, we achieved die counts of 0.88 (0.02), 0.88 (0.02), 0.90 (0.00), and 0.95 (0.02) on 3D U-Net, 3D Res U-Net, 3D Dense U-Net, 3D Wide U-Net, and 3D U-Net++, respectively.

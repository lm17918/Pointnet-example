# PointNet for 3D Object Classification with PyTorch
This is a PyTorch implementation of the PointNet network for 3D object classification, trained on a subset of the ModelNet dataset.
This repository is based on the project: https://www.kaggle.com/code/balraj98/pointnet-for-3d-object-classification-ii-pytorch/notebook

## Dataset
The ModelNet dataset is a large-scale 3D CAD model dataset, used for research in computer graphics, computer vision, and robotics. This dataset contains 3D models of objects from 40 different categories, with 55,984 unique CAD models in total.

The dataset can be found here: https://vision.princeton.edu/projects/2014/3DShapeNets/

## PointNet Architecture
PointNet is a deep neural network architecture designed for 3D object classification, segmentation, and point cloud processing tasks. The input to the PointNet network is a set of 3D points, represented as (x, y, z) coordinates in a point cloud.

The PointNet architecture consists of several layers of multi-layer perceptrons (MLPs), followed by a max-pooling layer and a global max-pooling layer. The output of the network is a feature vector that represents the input point cloud, which can be used for classification or other downstream tasks.

## Implementation
The implementation of PointNet for 3D object classification in this repository is written in PyTorch. It uses the Hydra and WandB libraries to allow easy management of parameters and experiments.

The code consists of the following files:

main.py: Main script for training and validation the PointNet network.
src/models.py: Definition of the PointNet network architecture.
src/dataset.py: Definition of the dataset class for loading the ModelNet data.
src/utils.py: Utility functions for training and validating the model.

## Training and validation
To train and validate the PointNet network, run the following command:

python main.py
This will train the model on the training set and validate it on the validation set. The results will be logged using the WandB library, allowing for easy visualization and comparison of experiments.

## Hyperparameter Search
The main.py script is set up to perform a grid search over multiple hyperparameters for the model and training. The hyperparameters are defined in the config/multirun/multirun.yaml file, and the script will train and validate the model for every combination of hyperparameters.

## Results
After training and validation the model, the results can be visualized using the WandB interface. The results include the accuracy of the model on the training and validation set, as well as the computed loss.

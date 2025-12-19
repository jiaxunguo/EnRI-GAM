# "Enhancing Rotation-Invariant 3D Learning with Global Pose Awareness and Attention Mechanisms."

## Dependencies
 
Python=3.7, 
CUDA=10.0,
PyTorch=1.4.0, 
torch_geometric=1.6.0,
torch_cluster=1.5.4,
torch_sparse=0.6.1,
torch_scatter=2.0.4,
tensorboardX, 
scikit-learn, 
numpy,
termcolor

## Data

First, please download the ModelNet40 dataset from [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)(1.6G), 
and place it at `data/modelnet40_normal_resampled`. 

## Point Cloud Classification on ModelNet40


To train a model under `SO(3)` or `z` rotations:

    python train_classification_modelnet40.py --trainset so3 --testset so3

To visualize the training process, please run:

    tensorboard --logdir log
 
## Acknowledgement
- The code framework is borrowed from [RISurConv](https://github.com/cszyzhang/RISurConv)
- The code for classification architecture is borrowed from [DGCNN](https://github.com/WangYueFt/dgcnn)

## TODO
Code on ScanObjectNN and ShapeNetPart.

##



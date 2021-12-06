# AdvGCGAN
 
The code repository for our paper Generating Unrestricted 3D Adversarial Point Clouds [arXiv](http://arxiv.org/abs/2111.08973)

# Installation
--
This repository is based on Python 3.8, Pytorch 1.8.1, CUDA 11.2 on Ubuntu 18.04.

1. Set up environments for the codes.

   ```shell
   pip install -r requirements.txt
   ```

2. (Optional) Install KNN_CUDA.

   ```shell
   pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
   ```

3. (Optional) Compile Structural_Losses operators by using the Makefile located at evaluation/pytorch_structural_losses

4. Download ShapeNetCore data set and unzip.

   ```shell
   wget -c https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0.zip
   unzip shapenetcore_partanno_segmentation_benchmark_v0.zip
   ```

5. Download the pretrained PointNet, PointNet++, and DGCNN model from [GoogleDrive](https://drive.google.com/drive/folders/1gdbQzLKFiXCMELI_e44YVHOlgaaoYYjd?usp=sharing), extract them to the folder `checkpoints/shapenetpart_pointnet_train/`, `checkpoints/shapenetpart_pointnet2_train/`, and `checkpoints/shapenetpart_dgcnn_train/`. 


# Usage
--

1. Parameter Settings:

   You should adjust the parameters for the training of AdvGCGNA, they can be found in arguments.py.
   You should specify the classes for the traning. In our paper, we use ['Airplane', 'Table','Car'] for the class_choice . We suggest only train with 2 classes due to the limited performance of the Generator. After the GAN training, you should adjust the class_choice to ['Table'] if you want to generate adversarial point clouds for 'Table' class.
 
2. Run:
   
   According to the paper, there are two stage training. Please use train.py for GAN training, and use train-attack.py for adversarial training.
   
   ```shell
   python train.py
   python train-attack.py
   ```

3. Evaluate

   We also provide the evaluation file to test the performance of different attack models. You should follow the two (Optioanl) steps for using our provided codes. The evaluation contains SRS and SOR defense methods. You also should adjust the parameters in arguments.py.
   
   ```shell
   python evaluate.py
   ```


# Acknowledgement

The basic structure of our codes is adopted from the paper [_3D Point Cloud Generative Adversarial Network Based on Tree Structured Graph Convolutions_](https://arxiv.org/abs/1905.06292) (Dong Wook Shu*, Sung Woo Park*, Junseok Kwon)


# Reference
--

Please cite our paper if you found any helpful information:


    @article{dai2021generating,
      title={Generating Unrestricted 3D Adversarial Point Clouds},
      author={Dai, Xuelong and Li, Yanjie and Dai, Hua and Xiao, Bin},
      journal={arXiv preprint arXiv:2111.08973},
      year={2021}
    }

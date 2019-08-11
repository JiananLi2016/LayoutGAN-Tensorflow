# LayoutGAN-Tensorflow
LayoutGAN in Tensorflow

Official Tensorflow implementation of "LayoutGAN: Generating Graphic Layouts with Wireframe Discriminators" publishsed in ICLR 2019: 
https://openreview.net/forum?id=HJxB5sRcFQ. 

Some codes are implemented from https://github.com/carpedm20/DCGAN-tensorflow. 


Prerequisites

Python 2.7
Tensorflow 1.2.0


Usage

First, download the trasformed point layout representation of MNIST dataset from:
https://drive.google.com/file/d/1x_cXGILI5EQvN7ksUH7Kz8-VCPsGnq_S/view?usp=sharing

To train a model with downloaded dataset:
bash ./experiments/scripts/train_mnist.sh

Results on MNIST
![alt text](https://drive.google.com/file/d/1xbbeficxACkzVukLDQLLsH5bVB8tYtbL/view?usp=sharing)


Author
Jianan Li

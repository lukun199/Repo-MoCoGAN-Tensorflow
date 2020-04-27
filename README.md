# Repo-MoCoGAN-Tensorflow

This is a **SIMPLE** Tensorflow implementation of CVPR18 Paper——MoCoGAN: Decomposing Motion and Content for Video Generation. 

This project does not obey all the details of the original paper (their idea of decoupling the latent code into content and motion is used) and **ONLY** tests a simple unconditional generation scenario in Weizmann dataset. Some problems (e.g., the loss of discriminator drops to a small value easily) have not been fixed. I hope this experiment could help those who need any reference.

The original implementation in Pytorch could be fonoud here(https://github.com/sergeytulyakov/mocogan), where the authors also provide some links to other implementations in Pytorch and Chain.

## Requirements:
+ Tensorflow 1.x  (in tf 2.x, adding `import tensorflow.compat.v1 as tf` and `tf.disable_v2_behavior()` also works)
+ cv2


Results:

## training:


## testing:


## how to run the code:

1. Download the dataset from the original repo.
2. Put the folder `action` into the root directory. cd to this directory.
3. python train.py

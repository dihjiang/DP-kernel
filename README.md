# DP-kernel

This code repo corresponds to the paper "Functional Renyi Differential Privacy for Generative Modeling" in NeurIPS 2023.

This repo is developed using Python 3.6 (but I believe it should also be compatible with higher versions of Python).

## Preparation
### 1. Tensorflow Privacy
We use this standalone library to compute (ε, δ)-DP of an algorithm given training parameters, regardless of what model you choose. This library does not affect running code in our repo, so you can ignore this one if you are not interested. For verification, you can follow [this link](https://github.com/tensorflow/privacy) to install their privacy library.  We installed it using Python 3.9.10. Run the following code (either in script or console):
```
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import * # (wait for 1-2 min)
# params for conditional training, the first two params define subsampling rate, the third one is training epochs (equivalent to 200 * 60000 / 60 = 200000 training iterations), the fourth one is noise multiplier, the fifth one is delta.
compute_dp_sgd_privacy_statement(60000, 60, 200, 0.60, 1e-5, False) # => eps=10
compute_dp_sgd_privacy_statement(60000, 60, 200, 1.95, 1e-5, False) # => eps=1
compute_dp_sgd_privacy_statement(60000, 60, 200, 8.0,  1e-5, False) # => eps=0.2

# params for parallel training
compute_dp_sgd_privacy_statement(6000, 60, 200, 1.00, 1e-5, False) # => eps=10
compute_dp_sgd_privacy_statement(6000, 60, 200, 5.75, 1e-5, False) # => eps=1
compute_dp_sgd_privacy_statement(6000, 60, 200, 25,   1e-5, False) # => eps=0.2
```

### 2. Pyvacy
[Pyvacy](https://github.com/ChrisWaites/pyvacy) (under Apache-2.0 license) is a PyTorch version of an older version of Tensorflow Privacy. We use their sampling method.

### 3. CelebA dataset
Torchvision cannot download [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) automatically. You need to manually download this dataset, then specify its path in datasets.py (line 39 and 40). You can start with MNIST and FMNIST for convenience.

### 4. Install necessary packages
Install necessary packages in requirements.txt.

## Running DP-kernel
### Training conditional model
```
# change --noise among [0.6, 1.95, 8.0] to target epsilon=[10, 1, 0.2].
python3 DPCondmmdG_kernel_prod.py --dataset mnist  --batch_size 60 --image_size 32 --nc 1  --nz 10 --max_iter 200002 --gpu_device 0 --experiment "mnist_DPCondmmdG_kernel_prod"  --sigma_list 1 2 4 8 16  --noise 0.6  --vis_step 50000  --lr 5e-5  --n_class 10

python3 DPCondmmdG_kernel_prod.py --dataset fmnist --batch_size 60 --image_size 32 --nc 1  --nz 10 --max_iter 200002 --gpu_device 0 --experiment "fmnist_DPCondmmdG_kernel_prod"  --sigma_list 1 2 4 8 16  --noise 0.60  --vis_step 50000  --lr 5e-5  --n_class 10

python3 DPCondmmdG_kernel_prod.py --dataset celeba --batch_size 60 --image_size 32 --nc 3  --nz 10 --max_iter 200002 --gpu_device 0 --experiment "celeba_DPCondmmdG_kernel_prod"  --sigma_list 1 2 4 8 16  --noise 8.0  --vis_step 50000  --lr 5e-5  --n_class 2
```

### Test conditional model
```
# Change dataset name, noise level, experiment name accordingly for other settings.
# Compute FID.
python3 test.py --dataset mnist  --batch_size 60 --image_size 32 --nc 1  --nz 10 --num_iters 200000 --gpu_device 0 --experiment "mnist_DPCondmmdG_kernel_prod"  --sigma_list 1 2 4 8 16  --noise 0.60  --compute_FID  --num_samples 60000
python3 -m pytorch_fid ./Imgs/mnist/True  ./Imgs/mnist/Gen/60000  --batch-size 100   # change dataset name in the path for other datasets

# use 5 different seeds for running ML classification tasks 5 times.
python3 test.py --dataset mnist --batch_size 60 --image_size 32 --nc 1  --nz 10 --num_iters 200000 --gpu_device 0 --experiment "mnist_DPCondmmdG_kernel_prod"  --sigma_list 1 2 4 8 16  --noise 0.60  --ML_ACC  --num_samples 60000  --seed 1
```


### Training parallel model
```
# change noise among [1.0, 5.75, 25.0] on MNISTs to target epsilon=[10, 1, 0.2].
# change noise among [0.6, 1.95, 8.0] on CelebA to target epsilon=[10, 1, 0.2].
# Also, change class number in --select CLASS_NUMBER (0-9 for MNIST and FMNIST, 0-1 for CelebA) to train unconditional model on each class. You can run them in parallel.
python3 DPmmdG_per_class.py --dataset mnist --batch_size 60 --image_size 32 --nc 1  --nz 10 --max_iter 20002 --gpu_device 0 --experiment "mnist_DPmmdG_per_class"  --sigma_list 1 2 4 8 16  --noise 1.0  --vis_step 5000  --lr 5e-5  --select 0

python3 DPmmdG_per_class.py --dataset fmnist --batch_size 60 --image_size 32 --nc 1  --nz 10 --max_iter 20002 --gpu_device 0 --experiment "fmnist_DPmmdG_per_class"  --sigma_list 1 2 4 8 16  --noise 1.0  --vis_step 5000  --lr 5e-5  --select 0

python3 DPmmdG_per_class.py --dataset celeba  --batch_size 30 --image_size 32 --nc 3  --nz 10 --max_iter 200002 --gpu_device 0 --experiment "celeba_DPmmdG_per_class"  --sigma_list 1 2 4 8 16  --noise 8.0  --vis_step 50000  --lr 5e-5  --select 0  --seed 1
```

### Test parallel model
```
# Change dataset name, noise level, experiment name accordingly for other settings.
# Compute FID.
python3 test_union.py --dataset mnist  --batch_size 60 --image_size 32 --nc 1  --nz 10 --num_iters 20000 --gpu_device 0 --experiment "mnist_DPmmdG_per_class"  --sigma_list 1 2 4 8 16  --noise 1.0  --n_class 10  --compute_FID  --num_samples 60000
python3 -m pytorch_fid ./Imgs/mnist/True  ./Imgs/mnist/Gen/60000  --batch-size 100   # change dataset name in the path for other datasets

# use 5 different seeds for running ML classification tasks 5 times.
python3 test_union.py --dataset mnist --dataroot ./data/mnist --batch_size 60 --image_size 32 --nc 1  --nz 10 --num_iters 20000 --gpu_device 0 --experiment "mnist_DPmmdG_per_class"  --sigma_list 1 2 4 8 16  --noise 1.0  --n_class 10  --ML_ACC  --num_samples 60000  --seed 1
```

## Acknowledgement
The MMD code and the generative neural network are mostly referenced from [MMD-GAN](https://github.com/OctoberChang/MMD-GAN).

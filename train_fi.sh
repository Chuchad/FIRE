#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,

# specify the real and fake test directories in '--real_train_dir' and '--fake_train_dir' respectively

python train.py --real_train_dir train/dalle3/0_real --fake_train_dir train/dalle3/1_fake --total_epochs 100 --mode fire --resume ckpt/imagenet_fire/model_0030.pt --save_dir ckpt/imagenet_fire_ft_on_fakeinversion_dalle3

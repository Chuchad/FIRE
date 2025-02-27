#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,

# specify the real and fake test directories in '--real_train_dir' and '--fake_train_dir' respectively

python train.py --real_train_dir train/imagenet/real --fake_train_dir train/imagenet/adm --total_epochs 100 --mode fire --save_dir ckpt/imagenet_fire

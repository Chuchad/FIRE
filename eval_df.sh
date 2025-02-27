#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,

# specify the real and fake test directories in '--real_test_dir' and '--fake_test_dir' respectively

python eval.py --real_test_dir test/lsun_bedroom/ --fake_test_dir test/lsun_bedroom/ --mode fire --ckpt ckpt/imagenet_fire/model_0030.pt --norm_layer instance
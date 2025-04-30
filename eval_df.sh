#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,

# specify the real and fake test directories in '--real_test_dir' and '--fake_test_dir' respectively

python eval.py --real_test_dir test/lsun_bedroom/ --fake_test_dir test/lsun_bedroom/ --mode fire --ckpt ckpt/xxx.pt
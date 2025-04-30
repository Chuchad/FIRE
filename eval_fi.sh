#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,

# no need to specify the real and fake test directories when testing on the self-collected dataset

python eval.py --real_test_dir 0_real --fake_test_dir 1_fake --mode fire --ckpt ckpt/xxx.pt

#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,

# no need to specify the real and fake test directories when testing on the self-collected dataset

python eval.py --real_test_dir 0_real --fake_test_dir 1_fake --mode fire --ckpt ckpt/imagenet_fire_ft_on_fakeinversion_kandinsky3/top_e082_0.056.pt --norm_layer instance --resize True

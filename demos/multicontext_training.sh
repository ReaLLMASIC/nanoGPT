#!/bin/bash


# TODO: prepare each of the datasets used for below

python3 train.py \
  --gns_type exact \
  --training_mode multicontext \
  --multicontext \
  --multicontext_datasets "shakespeare_char" "shakespeare_char_mobius" "shakespeare_char_order" "shakespeare_char_pos_2" \
  --vocab_sizes 65 256 24 16


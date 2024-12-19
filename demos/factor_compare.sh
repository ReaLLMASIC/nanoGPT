#!/bin/bash

# cd into main directory
cd ../

# Run comparison of factorization initializations

# ML factorization init
python3 train.py \
--init_from "gpt2" \
--out_dir out_gpt2_wte_customloss \
--dataset openwebtext \
--import_wte_npy  "100.0_best_wte.npy" \
--import_scale_matrices_npz  "100.0_best_scale_matrices.npz" \
--max_iters 20000 \
--n_embd_wte 100 \
--sample_each_eval \
--max_sample_tokens 60 \
--sample_start_tokens  $'Once upon a time, ' | tee "factor_compare_customloss_logs_20000_openwebtext.txt"

# Standard embedding random init (0.0 mean 0.02 stddev)
python3 train.py \
--init_from "gpt2" \
--out_dir out_gpt2_wte_random \
--dataset openwebtext \
--import_wte_npy  "./factorization_util/random_init/100_wte.npy" \
--import_scale_matrices_npz  "./factorization_util/random_init/100_scale_matrices.npz" \
--n_embd_wte 100 \
--max_iters 20000 \
--sample_each_eval \
--max_sample_tokens 60 \
--sample_start_tokens  $'Once upon a time, ' | tee "factor_compare_random_logs_20000_openwebtext.txt"


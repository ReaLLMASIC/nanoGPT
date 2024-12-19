#!/bin/bash

python3 run_factor.py  \
        --viz_owner "default" \
        --viz_studyid "0" \
        --matrix_path "gpt2_initial_wte.npy" \
        --lr_start "2e-3" \
        --lr_decay "linear" \
        --lr_stop "1e-3" \
        --num_seeds 3 \
        --num_epochs 10000 \
        --A_start 5 \
        --A_step 5 \
        --A_end 105 \
        --loss_fn "direction_magnitude" \
        --output_dir "out_gpt2_factor_sweep" \
        --output_csv "gpt2_factor_sweep.csv"

python3 run_factor.py  \
        --viz_owner "default" \
        --viz_studyid "1" \
        --lr_start "2e-3" \
        --lr_decay "linear" \
        --lr_stop "1e-3" \
        --num_seeds 3 \
        --num_epochs 10000 \
        --A_start 5 \
        --A_step 5 \
        --A_end 105 \
        --loss_fn "direction_magnitude" \
        --output_dir "out_random_factor_sweep" \
        --output_csv "random_matrix_factor_sweep.csv"


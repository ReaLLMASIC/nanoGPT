#!/bin/bash

cd ../

python3 sample.py \
    --device cuda \
    --no-print_model_info \
    --out_dir "out_gpt2_standard" \
    --start $'#B:\nHello, how are you doing? My major in college is languages.' \
    --num_samples 0 \
    --max_new_tokens 0 \
    --apply_to_layer_idx 6 \
    --save_avg_vector "en_vector.npy"

python3 sample.py \
    --device cuda \
    --no-print_model_info \
    --out_dir "out_gpt2_standard" \
    --start $'#B:\nHola, como estas? Estoy estudiando las lenguas en la universidade.' \
    --num_samples 0 \
    --max_new_tokens 0 \
    --apply_to_layer_idx 6 \
    --save_avg_vector "es_vector.npy"

for i in $(seq -2.25 0.5 2.25); do
python3 sample.py \
    --device cuda \
    --no-print_model_info \
    --out_dir "out_gpt2_standard" \
    --start $'\n#B:\n' \
    --apply_vector_file1 en_vector.npy \
    --apply_vector_file2 es_vector.npy \
    --apply_to_layer_idx 6 \
    --steering_vector_scaling_factor "$i" \
    --num_samples 1 \
    --max_new_tokens 60
done


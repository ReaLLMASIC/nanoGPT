#!/bin/bash

# Training fully quantized model
python3 train.py \
--out_dir "quantized_model" \
--n_layer "2" \
--n_head "2" \
--n_kv_group "2" \
--n_embd "60" \
--max_iters "100" \
--block_size "32" \
--eval_iters "50" \
--log_interval "20" \
--quantize_linear_method "symmetric_quant" \
--activations_quant_method "symmetric_quant" \
--dtype "bfloat16" \
--quantization_warmup_iters 0 \
--quantize_attn_act \
--quantize_mlp_act \
--linear_variant_attn "quantized_linear" \
--linear_variant_mlp "quantized_linear" \
--store_activations

# Save a model's quantized values to a .pkl file
python3 quantization/save_weights.py \
--out_dir "quantized_model" \
--file_name "quantized_data" \
--file_type "pkl"

# Create a histogram for every quantized weight and activation of a model
python3 quantization/visualize.py \
--file_name "quantized_data.pkl" \
--image_folder "quantized_images" \
--weight "all" \
--graph "histogram"
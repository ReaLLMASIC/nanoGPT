#!/bin/bash
# multicontext_demo.sh

set +x

pushd data/mnist
# obtain dataset
bash get_dataset.sh
# convert images to 16x16 ascii images
python3 gray2.py \
--image-dir mnist_images \
--output-dimensions 10x10 \
--levels 8 \
--number-placement before \
--append-to-file
# create mc datasets and folders and tokenize
bash split.sh
popd

python3 train.py \
  --training_mode multicontext \
  --multicontext \
  --use_flash_lobo \
  --use_flash_lobo_per_head \
  --flash_lobo_log_const 1.0 \
  --multicontext_datasets \
      mnist \
      mnist_abs \
      mnist_column \
      mnist_row \
   --max_iters 3000 \
   --use_rotary_embeddings \
   --no-use_abs_pos_embeddings \
   --out_dir ./out_mnist \
   --block_size 280 \
   --compile

python3 sample.py \
  --out_dir ./out_mnist \
  --multicontext \
  --multicontext_datasets \
      mnist \
      mnist_abs \
      mnist_column \
      mnist_row \
  --multicontext_start \
    "0" \
    "_" \
    "_" \
    "_" \
  --max_new_tokens 280 \
  --top_k 1 \
  --compile

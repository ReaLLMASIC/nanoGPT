#!/bin/bash
# multicontext_demo.sh

pushd data/shakespeare_char
bash get_dataset.sh
popd

pushd data/shakespeare_char_cvp/
bash get_dataset.sh
popd

pushd data/shakespeare_char_in_word_position/
bash get_dataset.sh
popd

pushd data/shakespeare_char_part_of_speech/
bash get_dataset.sh
popd

pushd data/shakespeare_char_since_newline/
bash get_dataset.sh
popd

python3 train.py \
  --training_mode multicontext \
  --multicontext \
  --use_flash_lobo \
  --use_flash_lobo_per_head \
  --multicontext_datasets \
      shakespeare_char \
      shakespeare_char_cvp \
      shakespeare_char_in_word_position \
      shakespeare_char_part_of_speech \
      shakespeare_char_since_newline \
   --vocab_sizes 65 4 65 21 67 \
   --max_iters 3000 \
   --dropout 0.2 \
   --use_rotary_embeddings \
   --no-use_abs_pos_embeddings \
   --wte_weight_tying \
   --out_dir ./fl_out_weight_tying \
   --compile

python3 sample.py \
  --out_dir ./fl_out_weight_tying \
  --multicontext \
  --multicontext_datasets \
    shakespeare_char \
    shakespeare_char_cvp \
    shakespeare_char_in_word_position \
    shakespeare_char_part_of_speech \
    shakespeare_char_since_newline \
  --multicontext_start \
    "Hello " \
    "21221_" \
    "12345_" \
    "aaaaa " \
    "123456" \
  --max_new_tokens 256 \
  --top_k 1 \
  --num_samples 1

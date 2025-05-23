#!/bin/bash

# Get current script directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# get into main directory
pushd "$script_dir/../" > /dev/null

# Using intermiediate datasets for tokenization
pushd "data/snac_cvja"
bash get_dataset.sh
popd

# running the training
python3 optimization_and_search/run_experiments.py --config explorations/stt_ipabytefallback.yaml --config_format yaml --output_dir snac_ipa_bf_ja_outs

# Doing the tiktoken tokenizer
pushd "data/snac_cvja"
python3 prepare.py -t ja_snac_text.txt --method tiktoken
popd

# running the training
python3 optimization_and_search/run_experiments.py --config explorations/stt_tiktoken.yaml --config_format yaml --output_dir snac_tiktoken_ja_outs

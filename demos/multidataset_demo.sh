#!/bin/bash

# Get current script directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# get into main directory
pushd "$script_dir/../" > /dev/null

# tokenize on commonvoice_zh
pushd "data/commonvoice_zh"
bash get_dataset.sh
popd

# tokenize on commonvoice_ko
pushd "data/commonvoice_ko"
bash get_dataset.sh
popd

# tokenize on commonvoice_ja
pushd "data/commonvoice_ja"
bash get_dataset.sh
popd

# running the training
python3 optimization_and_search/run_experiments.py --config explorations/multidataset.json --output_dir out_multi_zh
popd

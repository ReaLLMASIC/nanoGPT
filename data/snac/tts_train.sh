#!/bin/bash

# Set strict error handling
set -euo pipefail

# Get current script directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

echo "Running text to speech training..."

# Get the input file for running the dataset
python3 extract_data.py "tiny_sherlock_whisper_snac_combined" "input.txt" --directory

# prepare the dateset for training
python3 prepare.py -t input.txt --tokens_file tokens.txt --method custom

# running the training
pushd ../../
python3 run_experiments.py --config explorations/snac_text.json --output_dir out_test
popd


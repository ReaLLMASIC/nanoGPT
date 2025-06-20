#!/bin/bash

url="https://huggingface.co/datasets/JaepaX/korean_dataset"
out_dir="ko_dataset"

if [[ ! -d "${out_dir}" ]]; then
  mkdir -p "${out_dir}"
fi

for i in $(seq -f "%05g" 130 149); do
  wget -nc -O "${out_dir}/train_${i}.parquet" "${url}/resolve/main/data/train-${i}-of-00203.parquet?download=true"
done

# Extract information from the parquet files into json format
python3 parquet_to_json.py --dir ko_dataset

# Convert to snac and saved in json files
python3 utils/ko_to_snacipa.py "output_audio" "output.json" "korean_audio"

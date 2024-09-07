#!/bin/bash

### Instructions:
# 1. Replace "INSERT_URL_WITH_FILES" with the actual URL to the Parquet files.
# 2. Modify the "include_keys" array to specify the keys you want to include in the output.
# 3. (Optionally) Modify the "value_prefixes" array to set prefixes for each value, use "" for empty prefixes
# 4. Set "--skip_empty" to true if you want to skip empty fields, or false if not needed.
# 5. Set "--no_output_text" to true if you plan to process the intermediate json files in a custom manner.

# Run the Python script with the specified arguments

# Add url with dataset here:
mkdir json_outputs

wget -nc -O ./json_outputs/ko_ja_en.json https://huggingface.co/datasets/klei22/korean-english-jamon-parallel-corpora/resolve/main/korean_jamo_english.json?download=true

# uncomment and fill in if url has json datasets
python3 ./utils/get_json_dataset.py \
  --direct_json_input ./json_outputs/ko_ja_en.json \
  --include_keys "ko" "ph" "en" \
  --value_prefix $'\nko:' $'\nph:' $'\nen:' \
  --randomize_entries

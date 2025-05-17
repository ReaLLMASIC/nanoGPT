#!/usr/bin/env bash
# get_dataset.sh

set -e # exit early on any errors
set -x # print line before execution

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Download the tiny Shakespeare dataset from GitHub
wget -O input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

echo "Before conversion"
wc -c input.txt
python3 ./utils/char_convert.py input.txt --method cvp
echo "after conversion"
wc -c input.txt

python3 prepare.py --method char -t tokensfile.txt

cat tokensfile.txt

python3 prepare.py --method char -t input.txt --reuse_chars

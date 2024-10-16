#!/bin/bash
pip install huggingface_hub
huggingface-cli login
# huggingface.co/settings/tokens
huggingface-cli repo create repo_name --type {model, dataset, space}

git lfs install
git clone https://huggingface.co/username/repo_name
git add .
git commit -m "commit from $USER"
git push

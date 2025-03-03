# OLMo Fine-Tuning with Custom Attention

This script fine-tunes the pre-trained OLMo model using a custom attention variant. Currently, it supports the attention variants: clipped softmax, ReLU, normalized ReLU, gated softmax, softplus, normalized softplus, sigmoid, normalized sigmoid, off-by-one, and learned off-by-one softmax. It fine-tunes using the Wikitext-2 dataset for 20000 steps. 

## Usage

```bash
python3 OLMo_train.py --model_variant <variant> --lr <learning rate> --weight_decay <weight_decay> --max_steps <max training steps> --eval_steps <steps before evaluation> --save_steps <steps before saving model checkpoint>
```
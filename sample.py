import argparse
import json
import os
import pickle
from contextlib import nullcontext
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import tiktoken
from rich import print
from collections import OrderedDict
from torch.nn import functional as F

from model import GPT, GPTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Inference from trained models")
    parser.add_argument("--device", type=str, required=True, help="Device to run inference (e.g., 'cpu', 'cuda', 'cuda:0', 'cuda:1')")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to load checkpoint from")
    parser.add_argument("--quant_weights_file", type=str, default=None, help="File to export the quantized weights and scale factor")
    parser.add_argument("--visualize_weights_dir", type=str, default=None, help="Folder to save heatmaps of attention weights for all layers")
    parser.add_argument("--init_from", type=str, default="resume", help="Either 'resume' (from an out_dir) or a GPT-2 variant (e.g., 'gpt2-xl')")
    parser.add_argument("--start", type=str, default="\n", help="Start text for generation. Can specify a file using 'FILE:prompt.txt'")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of inference streams to draw")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Number of tokens to generate in each sample")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for predictions (1.0 = no change, < 1.0 = less random, > 1.0 = more random)")
    parser.add_argument("--top_k", type=int, default=200, help="Retain only the top_k most likely tokens")
    parser.add_argument("--seed", type=int, default=1337, help="Seed for pseudorandom number generator")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Torch data type for inference")
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, help="Compile the model (requires PyTorch 2.0)")
    parser.add_argument('--sample_file', type=str, default=None, help="Output file for inference")
    parser.add_argument('--interactive', action=argparse.BooleanOptionalAction, help="Enable interactive generation")
    parser.add_argument('--stop_string', type=str, default='~W', help="String to stop generation and allow user input")
    parser.add_argument('--show_heatmaps', action=argparse.BooleanOptionalAction, help="Show heatmaps of top-k choices for each token")
    parser.add_argument('--last_k_tokens', type=int, default=10, help="Number of last tokens to display in heatmaps")
    parser.add_argument('--chart_type', type=str, default='heatmap', choices=['heatmap', 'barchart'], help="Type of chart to display: 'heatmap' or 'barchart'")
    parser.add_argument('--block_size', type=int, default=None, help="Block size for context length, default is model's block size")
    parser.add_argument('--sym_rot_num_angles', type=int, default=None, help="Number of angles for symmetrical rotary embedding")
    return parser.parse_args()


def save_chart(probs, idx, decode, step, out_dir, last_k_tokens, chart_type, selected_token):
    top_k_probs, top_k_indices = torch.topk(probs, k=probs.size(-1))
    top_k_tokens = [decode([top_k_indices[0, i].item()]) for i in range(top_k_indices.size(1))]

    plt.figure(figsize=(10, 6))

    if chart_type == 'heatmap':
        sns.heatmap(top_k_probs.cpu().numpy().reshape(1, -1), annot=np.array(top_k_tokens).reshape(1, -1), fmt='', cmap='viridis')
    elif chart_type == 'barchart':
        colors = sns.color_palette('viridis', len(top_k_tokens))
        bars = plt.bar(top_k_tokens, top_k_probs.cpu().numpy().flatten(), color=colors)
        plt.xticks(rotation=90)
        for bar, token in zip(bars, top_k_tokens):
            if token == selected_token:
                bar.set_edgecolor('red')
                bar.set_linewidth(2)

    plt.title(f"Step {step}: Top-k Token Probabilities")
    last_tokens = decode(idx[0, -last_k_tokens:].tolist())
    plt.xlabel(f"Last {last_k_tokens} Tokens: {last_tokens}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{timestamp}_step{step}.png")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def interactive_generation(model, start_ids, device, max_new_tokens, temperature, top_k, stop_string, decode, encode):
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    while True:
        x, generated_text = model.generate_with_stop(x, max_new_tokens, stop_string, decode, temperature, top_k)
        print("[bold green]" + generated_text)

        user_input = input("User input (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        # Append the user input directly after the stop string
        x = torch.cat((x, torch.tensor(encode(user_input), dtype=torch.long, device=device)[None, ...]), dim=1)


def save_args(args, out_dir):
    with open(os.path.join(out_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

def save_quantized_weights(state_dict, out_file):
    to_save = OrderedDict()
    for k, v in list(state_dict.items()):
        if k.endswith("binarized_weight") or k.endswith("binarization_bias"):
            to_save[k] = v.cpu().numpy()
        if k.endswith("quantized_bias") or k.endswith("bias_norm") or k.endswith("quantized_weight") or k.endswith("weight_norm"):
            to_save[k] = v.cpu().numpy()

    with open(f"{out_file}.pkl", 'wb') as f:
        pickle.dump(to_save, f)

def visualize_weights(weights_dir, out_file, n_layers):
    filename = f"{out_file}.pkl"
    
    with open(filename, 'rb') as f:
        weights = pickle.load(f)
    print(weights.keys())
    #print(weights.values())
    for key, value in weights.items():
        print(key)
        print(value.shape)
    
    for i in range(n_layers):
        plt.rcParams["figure.figsize"] = [11, 3.5]
        plt.rcParams["figure.autolayout"] = True
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

        if f"transformer.h.{i}.attn.c_attn_q.binarized_weight" in weights:
            q_key = f"transformer.h.{i}.attn.c_attn_q.binarized_weight"
            k_key = f"transformer.h.{i}.attn.c_attn_k.binarized_weight"
            v_key = f"transformer.h.{i}.attn.c_attn_v.binarized_weight"
        if f"transformer.h.{i}.attn.c_attn_q.quantized_weight" in weights:
            q_key = f"transformer.h.{i}.attn.c_attn_q.quantized_weight"
            k_key = f"transformer.h.{i}.attn.c_attn_k.quantized_weight"
            v_key = f"transformer.h.{i}.attn.c_attn_v.quantized_weight"
        sns.heatmap(weights[q_key], ax=ax1)
        sns.heatmap(weights[k_key], ax=ax2)
        sns.heatmap(weights[v_key], ax=ax3)
        ax1.set_title(f"Heatmap of Query Weights for Layer {i}")
        ax2.set_title(f"Heatmap of Key Weights for Layer {i}")
        ax3.set_title(f"Heatmap of Value Weights for Layer {i}")
        #save to local dir
        #create a dir if it does not exist
        os.makedirs(weights_dir, exist_ok=True)
        plt.savefig(f"{weights_dir}/layer_{i}_weights.png")
    

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ptdtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    save_args(args, out_dir)

    if args.init_from == 'resume':
        ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        checkpoint['model_args']['dropout'] = 0.0
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        if args.quant_weights_file:
            save_quantized_weights(state_dict, args.quant_weights_file)

        model.load_state_dict(state_dict, strict=False)
    else:
        model = GPT.from_pretrained(args.init_from, dict(dropout=0.0))

    model.eval()
    model.to(args.device)
    if args.compile:
        model = torch.compile(model)

    if args.visualize_weights_dir:
        if not args.quant_weights_file:
            print("visualization requires weight file input")
            return
        visualize_weights(args.visualize_weights_dir, args.quant_weights_file, model.config.n_layer)

    if args.block_size:
        model.update_block_size(args.block_size)

    if args.sym_rot_num_angles:
        model.update_num_angles(args.sym_rot_num_angles)

    load_meta = False
    meta_path = None
    separator_token = None
    if args.init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:

        meta_paths = [
                os.path.join(args.out_dir, 'meta.pkl'),
                os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
                ]

        load_meta = False
        for meta_path in meta_paths:
            if os.path.exists(meta_path):
                load_meta = True
                break

    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        if 'tokenizer' in meta and meta['tokenizer'] == 'tiktoken':
            enc = tiktoken.get_encoding(meta['tiktoken_encoding'])
            print(f"using tiktoken encoding {meta['tiktoken_encoding']}")
            encode = lambda s: enc.encode(s, allowed_special={""})
            decode = lambda l: enc.decode(l)
        elif 'tokenizer' in meta and meta['tokenizer'] == 'sentencepiece':
            separator_token = "▁"
            stoi, itos = meta['stoi'], meta['itos']
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: ''.join([itos[i] for i in l])
        else:
            stoi, itos = meta['stoi'], meta['itos']
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: ''.join([itos[i] for i in l])

    if args.start.startswith('FILE:'):
        with open(args.start[5:], 'r', encoding='utf-8') as f:
            args.start = f.read()
    start_ids = encode(args.start)

    if args.interactive:
        interactive_generation(model, start_ids, args.device, args.max_new_tokens, args.temperature, args.top_k, args.stop_string, decode, encode)
    else:
        x = torch.tensor(start_ids, dtype=torch.long, device=args.device)[None, ...]

        # run generation
        with torch.no_grad():
            with ctx:
                for k in range(args.num_samples):
                    block_size = args.block_size if args.block_size else model.config.block_size
                    for step in range(args.max_new_tokens):
                        idx_cond = x if x.size(1) <= block_size else x[:, -block_size:]
                        logits, _ = model(idx_cond)
                        logits = logits[:, -1, :] / args.temperature
                        if args.top_k is not None:
                            v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                            logits[logits < v[:, [-1]]] = -float('Inf')
                        probs = F.softmax(logits, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                        x = torch.cat((x, idx_next), dim=1)

                        if args.show_heatmaps:
                            selected_token = decode([idx_next[0].item()])
                            save_chart(probs, x, decode, step, out_dir, args.last_k_tokens, args.chart_type, selected_token)

                    output_line = decode(x[0].tolist()).replace(separator_token, " ") if separator_token else decode(x[0].tolist())
                    print("[bold green]" + output_line)
                    print('---------------')
                    if args.sample_file:
                        with open(args.sample_file, "a") as file:
                            file.write(output_line)


if __name__ == "__main__":
    main()


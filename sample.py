import argparse
import json
import os
import pickle
import time
from contextlib import nullcontext
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import tiktoken
from rich import print
from torch.nn import functional as F
from collections import OrderedDict

from model import GPT, GPTConfig
from utils.model_info import print_summary, print_module_structure, print_model_blocks
from variations.model_variations import model_variation_dictionary

def parse_args():
    parser = argparse.ArgumentParser(description="Inference from trained models")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference (e.g., 'cpu', 'cuda', 'cuda:0', 'cuda:1')")
    parser.add_argument("--out_dir", type=str, default="out", help="Directory to load checkpoint from")
    parser.add_argument("--quantization_data_file", type=str, default=None, help="File name to export the quantized weights/activations, scale factor, and zero point")
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
    parser.add_argument('--rope_length', type=int, default=None, help="Number of embeddings to rotate (must be an even number <= total embedding size)")
    parser.add_argument('--token_boundary', type=str, default=None, help="optional separator between emitted tokens")
    parser.add_argument('--print_model_info', default=True, action=argparse.BooleanOptionalAction, help="print info about model before infernece")

    # Steering Vector Related
    parser.add_argument('--save_avg_vector', type=str, default=None, help="Path to save the average vector of the start text to an .npy file")
    parser.add_argument('--apply_vector_file1', type=str, default=None, help="First .npy file to load the vector for subtraction")
    parser.add_argument('--apply_vector_file2', type=str, default=None, help="Second .npy file to load the vector for subtraction")
    parser.add_argument('--steering_vector_scaling_factor', type=float, default=1.0, help="Scaling factor to apply after subtracting vectors")
    parser.add_argument('--apply_to_layer_idx', type=int, default=None, help="Layer index at which to apply the resulting vector")

    # Leanred Steering Vector Related
    parser.add_argument('--use_lsv', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--lsv_size',  type=int, default=1, help="Number of vectors to test")
    parser.add_argument('--lsv_scaling_factor',  type=float, default=None, help="scaling factor")
    parser.add_argument('--lsv_mixture',  type=float, nargs='+', default=None, help="scaling factor mixture")

    parser.add_argument("--eval_only", action=argparse.BooleanOptionalAction, help="Enable evaluation only mode to calculate and print validation loss")
    parser.add_argument("--eval_iters", type=int, default=250, help="iterations for evaluation")
    parser.add_argument("--eval_dataset", type=str, default=None, help="dataset for evaluation")

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


#TODO: Rename to reflect general purpose
def save_quantized_data(state_dict, out_file):
    to_save = OrderedDict()
    for k, v in list(state_dict.items()):
        # if "mlp_act" in k or "attn_act" in k or k.endswith("quantized_bias") or k.endswith("bias_norm") or k.endswith("zero_point") or k.endswith("quantized_weight") or k.endswith("weight_norm"):
        to_save[k] = v.cpu().numpy()

    with open(f"{out_file}.pkl", 'wb') as f:
        pickle.dump(to_save, f)

def load_validation_data(block_size, eval_dataset):
    # Load validation data similar to how train data is handled
    val_path = os.path.join('data', eval_dataset, 'val.bin')
    assert os.path.exists(val_path), f"Validation data file {val_path} not found."
    # Assuming validation data is similar in format to train data
    val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
    return val_data

def get_batch(data, block_size, device):
    # Create a random batch from the dataset
    ix = torch.randint(len(data) - block_size, (1,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

def calculate_validation_loss(model, val_data, block_size, eval_iters, device, dtype):
    model.eval()
    losses = []
    with torch.no_grad():
        total_time = 0
        for _ in range(eval_iters):
            X, Y = get_batch(val_data,  block_size, device)
            with torch.amp.autocast(device_type=device, dtype=dtype):
                start = time.perf_counter()
                logits, loss = model(X, Y)
                end = time.perf_counter()
                total_time += (end - start)
            losses.append(loss.item())
    print(f"Elapsed time: {total_time} seconds")
    return np.mean(losses)

def custom_char_with_byte_fallback_encode(text, stoi):
    ids = []
    for ch in text:
        if ch in stoi:
            ids.append(stoi[ch])
        else:
            # Byte fallback
            byte_sequence = ch.encode('utf-8')
            for byte in byte_sequence:
                ids.append(stoi[byte])
    return ids

def custom_char_with_byte_fallback_decode(ids, itos, custom_char_count):
    chars = []
    idx = 0
    while idx < len(ids):
        id = ids[idx]
        if id < custom_char_count:
            # It's a custom character
            chars.append(itos[id])
            idx += 1
        else:
            # It's a byte
            byte_buffer = []
            while idx < len(ids) and ids[idx] >= custom_char_count:
                byte_value = itos[ids[idx]]
                byte_buffer.append(byte_value)
                idx += 1
            # Convert byte buffer to character
            byte_array = bytes(byte_buffer)
            try:
                chars.append(byte_array.decode('utf-8'))
            except UnicodeDecodeError:
                chars.append('�')  # Replacement character for invalid sequences
    return ''.join(chars)

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
        if args.save_avg_vector:
            print(f"saving {args.save_avg_vector}")
            checkpoint['model_args']['obtain_vector_at_layer_idx'] = args.apply_to_layer_idx
            checkpoint['model_args']['obtain_vector_file'] = args.save_avg_vector
        # If vectors are provided, load and subtract them, then apply to a designated layer during generation
        if args.apply_vector_file1 and args.apply_vector_file2:
            vector1 = np.load(args.apply_vector_file1)
            vector2 = np.load(args.apply_vector_file2)
            diff_vector = vector1 - vector2
            torch.from_numpy(diff_vector).float().to(args.device)
            diff_vector_tensor = torch.from_numpy(diff_vector).float().to(args.device)
            diff_vector_cpu = diff_vector_tensor.cpu().numpy()  # Move the tensor to CPU and convert it to a NumPy array
            np.save("temp.npy", diff_vector_cpu)

            # Convert to tensor and set in the model for application at the designated layer
            checkpoint['model_args']['apply_vector_file']= "temp.npy"
            checkpoint['model_args']['apply_vector_at_layer_idx']= args.apply_to_layer_idx
            checkpoint['model_args']['apply_vector_scaling_factor']= args.steering_vector_scaling_factor
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        if args.quantization_data_file:
            save_quantized_data(state_dict, args.quantization_data_file)

        model.load_state_dict(state_dict, strict=False)

    else:
        # Need to create a completely "default" GPTConfig and overwrite using model_variations
        gptconf = GPTConfig()
        variation_dict = model_variation_dictionary[args.init_from]
        for k in variation_dict:
            gptconf[k] = variation_dict[k]
        model = GPT.from_pretrained(gptconf, model_type=args.init_from)

    # Load meta information if available
    load_meta = False
    meta_path = None
    separator_token = None
    if args.init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_paths = [
            os.path.join(args.out_dir, 'meta.pkl'),
            os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        ]

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
        elif 'tokenizer' in meta and meta['tokenizer'] == 'custom_char_with_byte_fallback':
            stoi = meta['stoi']
            itos = meta['itos']
            custom_char_count = meta['custom_char_count']
            encode = lambda s: custom_char_with_byte_fallback_encode(s, stoi)
            decode = lambda l: custom_char_with_byte_fallback_decode(l, itos, custom_char_count)
            print("Using CustomCharTokenizerWithByteFallback tokenizer")
        elif args.token_boundary:
            stoi, itos = meta['stoi'], meta['itos']
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: args.token_boundary.join([itos[i] for i in l])
        else:
            stoi, itos = meta['stoi'], meta['itos']
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: ''.join([itos[i] for i in l])


    if args.start.startswith('FILE:'):
        with open(args.start[5:], 'r', encoding='utf-8') as f:
            args.start = f.read()

    start_ids = encode(args.start)
    model.eval()
    model.to(args.device)

    # Print the model summary
    if args.print_model_info:
        print_summary(model)
        print_model_blocks(model)
        print_module_structure(model)


    if args.compile:
        model = torch.compile(model)

    # Inference with different block size (note: for this one cannot use abs pos embeddings)
    if args.block_size:
        model.update_block_size(args.block_size)

    # Inference with different number of angles
    if args.sym_rot_num_angles:
        model.update_num_angles(args.sym_rot_num_angles)

    # Inference with different Rope Length
    if args.rope_length:
        model.update_rope_length(args.rope_length)

    if args.eval_only:
        print("Running in eval_only mode...")
        # Load the validation dataset
        print(model.config.block_size)
        val_data = load_validation_data(model.config.block_size,
                                        args.eval_dataset)
        # Calculate validation loss
        val_loss = calculate_validation_loss(model, val_data,
                                             model.config.block_size,
                                             args.eval_iters, args.device, ptdtype)
        print(f"Validation Loss: {val_loss:.4f}")
        return

    x = torch.tensor(start_ids, dtype=torch.long, device=args.device)[None, ...]
    # Obtain vector from the specified layer and save it to a file if required
    if args.save_avg_vector:
        x = torch.tensor(start_ids, dtype=torch.long, device=args.device)[None, ...]
        # Run the model to trigger vector extraction
        with torch.no_grad():
            with ctx:
                block_size = args.block_size if args.block_size else model.config.block_size
                idx_cond = x if x.size(1) <= block_size else x[:, -block_size:]
                logits, _ = model(idx_cond)
        print(f"Obtained vector saved to {args.save_avg_vector}")



    if args.interactive:
        interactive_generation(model, start_ids, args.device, args.max_new_tokens, args.temperature, args.top_k, args.stop_string, decode, encode)
    else:
        # Run generation
        with torch.no_grad():
            with ctx:
                for k in range(args.num_samples):
                    if args.use_lsv:
                        model.set_lsv_index(k % args.lsv_size)
                        print("vector", k % args.lsv_size)
                        if args.lsv_scaling_factor is not None:
                            model.set_lsv_scaling_factor(args.lsv_scaling_factor)
                        if args.lsv_mixture is not None:
                            model.set_lsv_mode(2)
                            model.set_lsv_mixture(args.lsv_mixture)
                        else:
                            model.set_lsv_mode(1)
                    x = torch.tensor(start_ids, dtype=torch.long, device=args.device)[None, ...]
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
                    if args.apply_vector_file1:
                        print(f"Scaling factor: {args.steering_vector_scaling_factor}")
                    print("[bold green]" + output_line)
                    print('---------------')
                    if args.sample_file:
                        with open(args.sample_file, "w") as file:
                            file.write(output_line)

if __name__ == "__main__":
    main()


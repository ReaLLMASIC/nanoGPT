import argparse
from contextlib import nullcontext
import csv
import json
import math
import os
import random
import pickle
import shutil
import sys
import time

from model_info_util.model_info import print_summary, print_module_structure, print_model_blocks, print_model_tree
from monitoring_util.gpu_monitoring import get_gpu_memory_info

from rich.progress import Progress

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from statistics_util.statistic_plots import (
    initialize_statistics,
    plot_statistics,
    create_statistics,
)
from variations.model_variations import model_variation_dictionary

from model import GPT, GPTConfig

# Inference related imports
import tiktoken

def parse_args():

    parser = argparse.ArgumentParser()

    # argparse groups
    model_group = parser.add_argument_group('model_group')
    training_group = parser.add_argument_group('training_group')
    logging_group = parser.add_argument_group('logging_group')

    # Export Args
    ## Factored WTE
    model_group.add_argument('--import_wte_npy', default=None, type=str, help='Path to import the embedding table as a .npy file')
    model_group.add_argument('--export_wte_npy', default=None, type=str, help='Path to export the embedding table as a .npy file')
    model_group.add_argument('--export_wte_each_eval', default=False, action=argparse.BooleanOptionalAction, help="Requires --export_wte is not None. If this is so, will always export embedding to numpy after evaluation")
    model_group.add_argument('--import_wte_freeze', default=False, action=argparse.BooleanOptionalAction, help="Whether to freeze an imported wte")

    ## Factored Scale Matrices
    model_group.add_argument('--import_scale_matrices_npz', default=None, type=str, help='Path to import the scale matrices as a .npz file')
    model_group.add_argument('--export_scale_matrices_npz', default=None, type=str, help='Path to export the scale matrices as a .npz file')
    model_group.add_argument('--export_scale_matrices_each_eval', default=False, action=argparse.BooleanOptionalAction, help="Requires --export_scale_matrices_npz is not None. If this is so, will always export to npz after evaluation")
    model_group.add_argument('--import_scale_matrices_freeze', default=False, action=argparse.BooleanOptionalAction, help="Whether to freeze scaled_matrices")

    # I/O args
    training_group.add_argument('--out_dir', default='out', type=str)
    training_group.add_argument('--eval_interval', default=250, type=int)
    training_group.add_argument('--log_interval', default=10, type=int)
    training_group.add_argument('--eval_iters', default=200, type=int)
    training_group.add_argument('--eval_only', default=False, action=argparse.BooleanOptionalAction)

    # Loss variations
    training_group.add_argument('--focus_on_top1_loss', default=False, action=argparse.BooleanOptionalAction)

    # Sample args
    training_group.add_argument('--max_sample_tokens', default=None, type=int, help="If set, maximum number of tokens to sample and print after each validation loss")
    training_group.add_argument('--sample_each_eval', default=False, action=argparse.BooleanOptionalAction, help="Produce sample even if the validation loss did not improve. Allows for testing what overtraining looks like.")
    training_group.add_argument('--sample_start_tokens', default='\n', type=str)
    training_group.add_argument('--sample_only', default=False, action=argparse.BooleanOptionalAction, help="Run only the sampling process and exit")

    # Checkpoint args
    training_group.add_argument('--only_save_checkpoint_at_end', default=False, action=argparse.BooleanOptionalAction)
    training_group.add_argument('--always_save_checkpoint', default=False, action=argparse.BooleanOptionalAction)
    training_group.add_argument('--patience', default=None, type=int, help="if set, will stop training if the number of evaluations since val loss was seen to decrease exceeds 'patience' setting.")
    training_group.add_argument('--init_from', default='scratch', choices=['scratch', 'prev_run', 'resume', 'gpt2'], type=str)
    training_group.add_argument('--gpt2_type', default='gpt2', type=str)
    training_group.add_argument('--prev_run_ckpt', default='', type=str)
    training_group.add_argument('--csv_ckpt_dir', default='', type=str)

    # Data args
    training_group.add_argument('--dataset', default='shakespeare_char', type=str)
    training_group.add_argument('--batch_size', default=64, type=int)
    training_group.add_argument("--seed", default=1337, type=int)

    # Add a new argument for specifying multiple datasets
    training_group.add_argument('--dataset_list', default=None, nargs='+', type=str, help="If not None, training will be done from a list of datasets to train on, e.g. --dataset_list shakespeare wikitext103 openwebtext")
    training_group.add_argument('--dataset_interleaving', default=False, action=argparse.BooleanOptionalAction)
    training_group.add_argument('--dataset_interleaving_shuffle', default=False, action=argparse.BooleanOptionalAction)
    training_group.add_argument('--dataset_sampling_learning_rate', default=None, nargs='+', type=float, help="Sampling learning rates for each dataset in dataset_list.")
    training_group.add_argument('--dataset_sampling_probs', default=None, nargs='+', type=float, help="Sampling proportions for each dataset in dataset_list. Probabilities normally but proportions in dataset_interleaving")
    training_group.add_argument('--dataset_sampling_probs_final', default=None, nargs='+', type=float, help="If, set final sampling probabilities for each dataset in dataset_list.")
    training_group.add_argument('--dataset_sampling_probs_transition_method', default=None, type=str, choices=["linear", "cosine", "exponential"])


    # Model args
    model_group.add_argument('--block_size', default=256, type=int)
    model_group.add_argument('--n_layer', default=6, type=int)
    model_group.add_argument('--n_head', default=6, type=int)
    model_group.add_argument('--n_kv_group', default=None, type=int)
    model_group.add_argument('--n_embd', default=384, type=int, help="Size of embeddings in decoder layer and wte unless n_embd_wte is set." )
    model_group.add_argument('--n_embd_wte', default=None, type=int, help="If different from n_embd, an adapter table will be automatically created")
    model_group.add_argument('--n_embd_wte_scale_tying', default=True, action=argparse.BooleanOptionalAction, help="Enable weight tying for scale up and scale down matrices, only has effects if n_embd_wte is not 'None'.")
    model_group.add_argument('--dropout', default=0.2, type=float)
    model_group.add_argument('--use_post_ln', default=False, action=argparse.BooleanOptionalAction)
    model_group.add_argument('--window_size', default=None, type=int, help="Sliding window size, note this cannot be greater than block size")
    model_group.add_argument('--gate', default=False, action=argparse.BooleanOptionalAction, help="option for gated attention see https://arxiv.org/abs/2306.12929")
    model_group.add_argument('--use_moe', default=False,  action=argparse.BooleanOptionalAction, help="option for Mixture of Experts (MoE) architecture")
    model_group.add_argument('--moe_layer_freq', default=2, type=int, help="set frequency for replacing FFNs with MoE layers")
    model_group.add_argument('--n_experts', default=8, type=int, help="set number of experts per MoE layer")
    model_group.add_argument('--moe_top_k', default=2, type=int)
    model_group.add_argument('--moe_router_scheme', default="softmax", type=str, help="option to set routing scheme for MoE layer, defaults to softmax")


    ## Manual Steering Vector Options

    ### Applying Steering Vectors
    model_group.add_argument('--apply_vector_at_layer_idx', default=None, type=int)
    model_group.add_argument("--apply_vector_file", type=str, default=None, help="single vector to apply with scaling factor")
    model_group.add_argument("--apply_vector_scaling_factor", type=float, default=1.0, help="scaling factor to apply to steering vector")

    ### Options for intercepting and obtaining vectors
    model_group.add_argument('--obtain_vector_at_layer_idx', default=None, type=int)
    model_group.add_argument("--obtain_vector_file", type=str, default=None, help="initial KAN activation")

    ## Learned Steering Vector (LSV) Options
    lsv_variations = [
            "one_hot",
            "linear_comb",
            "one_hot_mlp",
            "ohmg",
            "ohmt",
            "ohmm",
            "ohma",
            "ohmgu",
            "ohmh",
            "mol",
        ]
    model_group.add_argument("--use_lsv", default=False, action=argparse.BooleanOptionalAction, help="whether to use Learned Steering Vectors")
    model_group.add_argument("--lsv_index", default=None, type=int, help="Which steering vector to use")
    model_group.add_argument("--lsv_variant", default="one_hot", type=str, choices=lsv_variations, help="Which steering vector to use")
    model_group.add_argument('--apply_lsv_at_layer_idx', default=None, type=int)

    training_group.add_argument("--lsv_focused_training", default=False, action=argparse.BooleanOptionalAction, help="train but only unfreeze lsv")

    ## MLP Options
    model_group.add_argument('--use_parallel_mlp', default=False, action=argparse.BooleanOptionalAction)
    model_group.add_argument("--mlp_variant", type=str, default="mlp", choices=["mlp", "kan", "swiglu"], help="MLP variation type")
    model_group.add_argument("--mlp_expansion_factor", type=int, default=4, help="If MLP like variant is used, set the expansion factor for the linear transformations, default is 4.")

    ## KAN Options
    model_group.add_argument("--kan_poly_order", type=int, default=3, help="Order of KAN non-linearity")
    model_group.add_argument("--kan_base_activation", type=str, default="silu", help="initial KAN activation")
    model_group.add_argument("--kan_middle_layers", type=int, nargs='+', help="List of integers", default=[])

    # Shared Parameter Settings
    model_group.add_argument('--shared_mlp_size', default=1, type=int, help="every 'k' contiguous blocks of mlp are shared")
    model_group.add_argument('--shared_mlp_sym', default=False, action=argparse.BooleanOptionalAction)
    model_group.add_argument('--shared_attn_size', default=1, type=int, help="every 'k' contiguous blocks of attn are shared")
    model_group.add_argument('--shared_attn_sym', default=False, action=argparse.BooleanOptionalAction, help="symmetrical attention sharing")

    # NORM VARIATIONS
    model_group.add_argument("--norm_variant_attn", type=str, default="rmsnorm", choices=["krmsnorm", "prmsnorm", "rmsnorm", "layernorm"])
    model_group.add_argument("--norm_variant_output", type=str, default="rmsnorm", choices=["krmsnorm", "prmsnorm", "rmsnorm", "layernorm"])
    model_group.add_argument('--bias', default=False, action=argparse.BooleanOptionalAction, help="only used for layernorm variation option")
    model_group.add_argument("--prmsnorm_pct", default=0.0625, type=float, help="percentage (1 being 100 percent) of first entries used for partial rms" )
    model_group.add_argument("--krmsnorm_num", default=10, type=int, help="max number of first entries for partial rms" )
    model_group.add_argument("--krmsnorm_quantize_type", type=str, default="none", choices=["int8", "int16", "none"])
    model_group.add_argument('--krmsnorm_enable_gain', default=True, action=argparse.BooleanOptionalAction, help="include gain in kRMSNorm")
    model_group.add_argument("--krmsnorm_selection_type", type=str, default="last", choices=["first", "last", "random"])
    model_group.add_argument("--krmsnorm_recompute_percentage", type=float, default=None, help="percentage needed within the total RMS to not trigger recompute")

    activation_variations = [
            "celu",
            "elu",
            "gelu",
            "glu",
            "leaky_relu",
            "learned_spline",
            "mish",
            "piecewise",
            "pfla",
            "prelu",
            "relu",
            "relu6",
            "rrelu",
            "selu",
            "sigmoid",
            "silu",
            "softplus",
            "softsign",
            "squared_relu",
            "tanh",
        ]

    # ACTIVATION VARIATIONS
    model_group.add_argument( "--activation_variant", type=str, default="gelu", choices=activation_variations)
    model_group.add_argument( "--num_of_points", type=int, default="8")

    # LINEAR VARIATIONS
    linear_variants = ["linear", "bitlinear", "bitlinear_1p58", "bitlinear_optimized", "kan","quantized_linear"]
    model_group.add_argument("--linear_variant_attn", type=str, default="linear", choices=linear_variants)
    model_group.add_argument("--linear_variant_q", type=str, default=None, choices=linear_variants, help="sets the linear variant for c_attn_q in attention (takes precedence over linear_variant_attn)")
    model_group.add_argument("--linear_variant_k", type=str, default=None, choices=linear_variants, help="sets the linear variant for c_attn_k in attention (takes precedence over linear_variant_attn)")
    model_group.add_argument("--linear_variant_v", type=str, default=None, choices=linear_variants, help="sets the linear variant for c_attn_v in attention (takes precedence over linear_variant_attn)")
    model_group.add_argument("--linear_variant_attn_proj", type=str, default=None, choices=linear_variants, help="sets the linear variant for c_proj in attention (takes precedence over linear_variant_attn)")
    model_group.add_argument("--linear_variant_mlp", type=str, default="linear", choices=linear_variants)
    model_group.add_argument("--linear_variant_mlp_up", type=str, default=None, choices=linear_variants, help="sets the linear variant for c_fc in mlp (takes precedence over linear_variant_mlp)")
    model_group.add_argument("--linear_variant_mlp_down", type=str, default=None, choices=linear_variants, help="sets the linear variant for c_proj in mlp (takes precedence over linear_variant_mlp)")
    ## Linear Weight Initialization Options
    model_group.add_argument( "--linear_mean_init", type=float, default=0.0)
    model_group.add_argument( "--linear_std_init", type=float, default=0.02)

    # Quatization

    ## Quantization Method Options
    quant_methods = ["symmetric_quant", "affine_quant", "stochastic_quant"]

    ## WTE
    model_group.add_argument("--quantize_wte", default=None, action=argparse.BooleanOptionalAction, help="Whether the word embedding is quantized")
    model_group.add_argument("--quantize_wte_method", type=str, default="affine_quant", choices=quant_methods, help="function used for word embedding quantization")
    model_group.add_argument("--quantize_wte_bits", type=int, default=8, help="number of bits for word embedding quantization")

    ## WPE
    model_group.add_argument("--quantize_wpe", default=None, action=argparse.BooleanOptionalAction, help="Whether the word position embedding is quantized")
    model_group.add_argument("--quantize_wpe_method", type=str, default="affine_quant", choices=quant_methods, help="function used for position embedding quantization")
    model_group.add_argument("--quantize_wpe_bits", type=int, default=8, help="number of bits for position embedding quantization")

    ## Activations
    model_group.add_argument("--activations_quant_method", type=str, default="affine_quant", choices=quant_methods, help="function used for quantization of activations")

    ### Attention Activations
    model_group.add_argument("--quantize_attn_act", action=argparse.BooleanOptionalAction, default=False, help="quantize all input/output activations in attn")

    #### Whether to do Attention Activation quantization at the Arrow
    model_group.add_argument("--quantize_attn_act_input", action=argparse.BooleanOptionalAction, default=False, help="quantize input activation to attention")
    model_group.add_argument("--quantize_attn_act_qk_mult_q_input", action=argparse.BooleanOptionalAction, default=False, help="quantize query input activation to qk mult")
    model_group.add_argument("--quantize_attn_act_qk_mult_k_input", action=argparse.BooleanOptionalAction, default=False, help="quantize key input activation to qk mult")
    model_group.add_argument("--quantize_attn_act_softmax_input", action=argparse.BooleanOptionalAction, default=False, help="quantize input activation to softmax")
    model_group.add_argument("--quantize_attn_act_pv_mult_p_input", action=argparse.BooleanOptionalAction, default=False, help="quantize softmax input activation to pv mult")
    model_group.add_argument("--quantize_attn_act_pv_mult_v_input", action=argparse.BooleanOptionalAction, default=False, help="quantize value input activation to pv mult")
    model_group.add_argument("--quantize_attn_act_pv_mult_output", action=argparse.BooleanOptionalAction, default=False, help="quantize output activation of pv_mult")
    model_group.add_argument("--quantize_attn_act_output", action=argparse.BooleanOptionalAction, default=False, help="quantize output activation of attention")

    ### Default Precisions for Attention Activations
    model_group.add_argument("--quantize_attn_act_bits", type=int, default=8, help="number of bits for attn quantization")

    ### Overrides for granular Attention Activatinos
    model_group.add_argument("--quantize_attn_act_input_bits", type=int, default=None, help="number of bits for attention input quantization")
    model_group.add_argument("--quantize_attn_act_qk_mult_q_input_bits", type=int, default=None, help="number of bits for qk mult query input quantization")
    model_group.add_argument("--quantize_attn_act_qk_mult_k_input_bits", type=int, default=None, help="number of bits for qk mult key input quantization")
    model_group.add_argument("--quantize_attn_act_softmax_input_bits", type=int, default=None, help="number of bits for softmax input quantization")
    model_group.add_argument("--quantize_attn_act_pv_mult_p_input_bits", type=int, default=None, help="number of bits for pv mult softmax input quantization")
    model_group.add_argument("--quantize_attn_act_pv_mult_v_input_bits", type=int, default=None, help="number of bits for pv mult value input quantization")
    model_group.add_argument("--quantize_attn_act_pv_mult_output_bits", type=int, default=None, help="number of bits for pv mult output quantization")
    model_group.add_argument("--quantize_attn_act_output_bits", type=int, default=None, help="number of bits for attention output quantization")

    ### Whether to use MLP Activations
    model_group.add_argument("--quantize_mlp_act", action=argparse.BooleanOptionalAction, default=False, help="quantize all input/output activations in mlp")
    model_group.add_argument("--quantize_mlp_act_input", action=argparse.BooleanOptionalAction, default=False, help="quantize input activation to mlp")
    model_group.add_argument("--quantize_mlp_act_activation_input", action=argparse.BooleanOptionalAction, default=False, help="quantize input activation to activation function")
    model_group.add_argument("--quantize_mlp_act_activation_output", action=argparse.BooleanOptionalAction, default=False, help="quantize output activation of activation function")
    model_group.add_argument("--quantize_mlp_act_output", action=argparse.BooleanOptionalAction, default=False, help="quantize output activation of mlp")

    ### Default Precisions for MLP Activations
    model_group.add_argument("--quantize_mlp_act_bits", type=int, default=8, help="number of bits for mlp quantization")

    ### Overrides for granular MLP Activatinos
    model_group.add_argument("--quantize_mlp_act_input_bits", type=int, default=None, help="number of bits for mlp input quantization")
    model_group.add_argument("--quantize_mlp_act_activation_input_bits", type=int, default=None, help="number of bits for activation function input quantization")
    model_group.add_argument("--quantize_mlp_act_activation_output_bits", type=int, default=None, help="number of bits for activation function output quantization")
    model_group.add_argument("--quantize_mlp_act_output_bits", type=int, default=None, help="number of bits for mlp output quantization")

    ### Whether activations should be saved
    model_group.add_argument("--store_activations", action=argparse.BooleanOptionalAction, default=False, help="whether the activations should be saved as a buffer and updated through training")

    ## Linear Attn Weight Quantization Precision and Method

    ### Default methods and precisions
    model_group.add_argument("--quantize_linear_method", type=str, default="affine_quant", choices=quant_methods, help="function used for linear quantization")
    model_group.add_argument("--quantize_linear_bits", type=int, default=8, help="number of bits for linear quantization")

    #### Overrides for granular Methods and Precisions
    model_group.add_argument("--quantize_linear_attn_q_method", type=str, default=None, choices=quant_methods, help="function used for c_attn_q quantization")
    model_group.add_argument("--quantize_linear_attn_q_bits", type=int, default=None, help="number of bits for c_attn_q quantization")

    model_group.add_argument("--quantize_linear_attn_k_method", type=str, default=None, choices=quant_methods, help="function used for c_attn_k quantization")
    model_group.add_argument("--quantize_linear_attn_k_bits", type=int, default=None, help="number of bits for c_attn_k quantization")

    model_group.add_argument("--quantize_linear_attn_v_method", type=str, default=None, choices=quant_methods, help="function used for c_attn_v quantization")
    model_group.add_argument("--quantize_linear_attn_v_bits", type=int, default=None, help="number of bits for c_attn_v quantization")

    model_group.add_argument("--quantize_linear_attn_proj_method", type=str, default=None, choices=quant_methods, help="function used for c_proj in attention quantization")
    model_group.add_argument("--quantize_linear_attn_proj_bits", type=int, default=None, help="number of bits for c_proj in attention quantization")

    #### Overrides for Linear MLP Weight Quantization Precision and Method
    model_group.add_argument("--quantize_linear_mlp_up_method", type=str, default=None, choices=quant_methods, help="function used for mlp_up quantization")
    model_group.add_argument("--quantize_linear_mlp_up_bits", type=int, default=None, help="number of bits for mlp_up quantization")
    model_group.add_argument("--quantize_linear_mlp_down_method", type=str, default=None, choices=quant_methods, help="function used for mlp_down quantization")
    model_group.add_argument("--quantize_linear_mlp_down_bits", type=int, default=None, help="number of bits for mlp_down quantization")

    ## Quantized Linear Warmup Iterations -- how many to first use regular linear, before switching to quantized
    model_group.add_argument("--quantization_warmup_iters", type=int, default=100)

    # POSITIONAL EMBEDDING VARIATIONS
    model_group.add_argument('--use_rotary_embeddings', default=False, action=argparse.BooleanOptionalAction)
    model_group.add_argument('--sym_rot_num_angles', type=int, default=512, help="number of angles to use for symmetric rope variant")
    model_group.add_argument("--rope_variant", type=str, default="rope", choices=["rope", "soap"])
    model_group.add_argument("--rope_length", type=int, default=None, help="Defaults to all embeddings (if set to None), else must be even.")
    model_group.add_argument('--use_abs_pos_embeddings', default=True, action=argparse.BooleanOptionalAction)
    model_group.add_argument('--use_fire_embeddings', default=False, action=argparse.BooleanOptionalAction)
    model_group.add_argument('--shared_fire_embeddings', default=False, action=argparse.BooleanOptionalAction)

    ## Positional Embedding Weight Initialization Options
    model_group.add_argument( "--embedding_mean_init", type=float, default=0.0)
    model_group.add_argument( "--embedding_std_init", type=float, default=0.02)

    ## FIRE Options (Functional Interpolation for Relative Positional Encoding)
    model_group.add_argument( "--fire_log_bias", type=float, default=1.0, help="bias in the function psi(x) = log(cx + bias)")
    model_group.add_argument( "--fire_num_hidden_layers", type=int, default=1, help="number of hidden layers (sigmas) in mlp in FIRE without counting outermost sigma")
    model_group.add_argument( "--fire_mlp_width", type=int, default=32, help="mlp_width: one hidden dimension of linear layers in mlp in FIRE")
    model_group.add_argument( "--fire_init_c", type=float, default=0.1, help="init_c: initial value of log transformation parameter c in FIRE")
    model_group.add_argument( "--fire_init_L", type=float, default=512.0, help="init_L: initial value of threshold L in FIRE (fixed values without L_multiplier)")
    model_group.add_argument( "--fire_outermost_sigma", type=bool, default=False, action=argparse.BooleanOptionalAction, help="whether or not adding outermost sigma in mlp in FIRE")

    # SOFTMAX VARIATIONS
    softmax_variations = [
        "saturatingconsmax",
        "consmax",
        "consmax_v2",
        "consmax_quan",
        "polymax",
        "relumax",
        "relu2max",
        "sigmoidmax",
        "vpolymax",
        "exppolymax",
        "strongermax",
        "softermax",
        "sigsoftmax",
        "softmax",
        "softplus",
        "squareplus",
        "exppolymax",
        ]

    ## Selection of softmax variation for attention and output layers
    model_group.add_argument("--softmax_variant_attn", type=str, default="softmax", choices=softmax_variations)
    model_group.add_argument("--softmax_variant_output", type=str, default="softmax", choices=softmax_variations)
    model_group.add_argument("--disable_flash_attention", default=False, action=argparse.BooleanOptionalAction, help="manual setting to disable flash attention")

    ## Custom Softmax Variation Options
    ### ConSmax and SaturatingConSmax Options
    model_group.add_argument("--consmax_initial_beta", type=float, default=2.5)
    model_group.add_argument("--consmax_initial_gamma", type=float, default=100.0)
    model_group.add_argument('--consmax_use_euler_base', default=True, action=argparse.BooleanOptionalAction)
    model_group.add_argument("--consmax_base", type=float, default=2.0)

    ### Special Options for ConSmaxV2
    model_group.add_argument("--consmax_per_head", default=True, action=argparse.BooleanOptionalAction)
    model_group.add_argument("--consmax_v2_clamping", default=False, action=argparse.BooleanOptionalAction)
    model_group.add_argument("--consmax_v2_clamp_value", type=float, default=80.0, help="maximum value to clamp inputs")

    ### Special Options for SaturatingConSmax
    model_group.add_argument("--consmax_saturation", type=float, default=11.0, help="point where we transition from consmax to linear saturatingconsmax, defaults to 11 to approximate e^x sat for fp16")
    model_group.add_argument('--consmax_learnable_beta', default=True, action=argparse.BooleanOptionalAction)
    model_group.add_argument('--consmax_learnable_gamma', default=True, action=argparse.BooleanOptionalAction)

    ### Polymax Options
    model_group.add_argument("--polymax_x_intercept", type=float, default=-100.0)
    model_group.add_argument("--polymax_y_intercept", type=float, default=1.0)
    model_group.add_argument("--polymax_power", type=float, default=2.0)
    model_group.add_argument("--polymax_divisor", type=float, default=1000.0)

    ### ReLUMax Options
    model_group.add_argument("--relumax_divisor", type=float, default=256.0)

    ### ReLU2Max Options
    model_group.add_argument("--relu2max_divisor", type=float, default=256.0)

    ### SimgoidMax Options
    model_group.add_argument("--sigmoidmax_divisor", type=float, default=256.0)

    ### SigSoftmax Options
    model_group.add_argument('--sigsoftmax_use_euler_base', default=True, action=argparse.BooleanOptionalAction)
    model_group.add_argument("--sigsoftmax_base", type=float, default=2.0)

    ### Strongermax Options - Testing Incremental Adjustments to Regular Softmax
    model_group.add_argument("--strongermax_strength", type=float, default=2.718)
    model_group.add_argument('--strongermax_sum_to_1', default=True, action=argparse.BooleanOptionalAction)
    model_group.add_argument("--strongermax_divisor", type=float, default=1.0)
    model_group.add_argument('--strongermax_use_xmax', default=True, action=argparse.BooleanOptionalAction)
    model_group.add_argument('--strongermax_xmax_guess', type=float, default=None)
    model_group.add_argument('--strongermax_overflow_recompute', default=False, action=argparse.BooleanOptionalAction)

    ### ExpPolymax Options
    model_group.add_argument('--exppolymax_use_euler_base', default=True, action=argparse.BooleanOptionalAction)
    model_group.add_argument("--exppolymax_base", type=float, default=4.0)
    model_group.add_argument("--exppolymax_y_intercept", type=float, default=1.0)
    model_group.add_argument("--exppolymax_power", type=float, default=2.0)
    model_group.add_argument("--exppolymax_divisor", type=float, default=1000.0)

    ### Softermax Specific Options
    model_group.add_argument('--softermax_use_xmax', default=True, action=argparse.BooleanOptionalAction)

    ### SoftPlus Options
    model_group.add_argument('--softplus_divisor', type=float,default=100.0)
    ### SquarePlus Options
    model_group.add_argument('--squareplus_divisor', type=float,default=100.0)

    ### Sequence Length Division https://arxiv.org/abs/2309.
    model_group.add_argument('--div_by_seq_len', default=False, action=argparse.BooleanOptionalAction)

    # Gradient Checkpointing
    model_group.add_argument('--use_gradient_checkpointing', default=False, action=argparse.BooleanOptionalAction, help="Memory efficient training, but takes longer time to train due to trading compute time for memory efficiency. For best memory tradeoff omit the --compile flag. For medium memory tradeoff add --compile.")
    model_group.add_argument('--recompute_backward_pass', default=False, action=argparse.BooleanOptionalAction, help="Recomputes for the backward pass, must use with --use_gradient_checkpointing")

    # Optimizer args
    training_group.add_argument('--max_iters', default=3500, type=int)
    training_group.add_argument('--weight_decay', default=1e-1, type=float)
    training_group.add_argument('--beta1', default=0.9, type=float)
    training_group.add_argument('--beta2', default=0.99, type=float)
    training_group.add_argument('--grad_clip', default=1.0, type=float)

    # LR schedule args
    training_group.add_argument('--learning_rate', default=1e-3, type=float)
    training_group.add_argument('--min_lr', default=1e-4, type=float)
    training_group.add_argument('--decay_lr', default=False, action=argparse.BooleanOptionalAction)
    training_group.add_argument('--lr_decay_iters', default=3500, type=int)
    training_group.add_argument('--lr_decay_match_max_iters', default=True, action=argparse.BooleanOptionalAction)
    training_group.add_argument('--warmup_iters', default=100, type=int)

    # DDP args
    training_group.add_argument('--backend', default='nccl', type=str)
    training_group.add_argument('--gradient_accumulation_steps', default=1, type=int)

    # System args
    training_group.add_argument('--device', default='cuda', type=str)
    training_group.add_argument("--dtype", type=str, default="float16", choices=["bfloat16", "float16", "float32"], help="torch data type for inference, e.g. 'int8'")
    training_group.add_argument('--compile', default=False, action=argparse.BooleanOptionalAction)

    # Logging args
    logging_group.add_argument('--log_project', default='out-test', type=str)
    logging_group.add_argument('--log_run_name', default='logs-test', type=str)
    logging_group.add_argument('--timestamp', default='', type=str)
    logging_group.add_argument('--save_nan_checkpoint', default=False, action=argparse.BooleanOptionalAction)

    # Module And Parameter Logging and Plots of Summary Statistics
    model_group.add_argument('--softmax_io_logging', default=False, action=argparse.BooleanOptionalAction, help="logs inputs and outputs of supported softmaxes")
    model_group.add_argument('--softmax_io_log_interval', default=1, type=int)
    model_group.add_argument('--consmax_beta_gamma_logging', default=False, action=argparse.BooleanOptionalAction, help="logs beta and gamma")
    logging_group.add_argument('--create_statistics', default=False, action=argparse.BooleanOptionalAction)
    logging_group.add_argument('--plot_statistics', default=False, action=argparse.BooleanOptionalAction)

    # CSV logging
    logging_group.add_argument('--csv_log', default=True, action=argparse.BooleanOptionalAction)
    logging_group.add_argument('--csv_dir', default='csv_logs', type=str)
    logging_group.add_argument('--csv_name', default='output', type=str, help="Output csv basename. Note, the .csv will be automatically appended.")

    # Tensorboard args
    logging_group.add_argument('--tensorboard_log', default=True, action=argparse.BooleanOptionalAction)
    logging_group.add_argument('--tensorboard_log_dir', type=str, default='logs')
    logging_group.add_argument('--tensorboard_run_name', type=str, default='logs-test')

    # Wandb args
    logging_group.add_argument('--wandb_log', default=False, action=argparse.BooleanOptionalAction)
    logging_group.add_argument('--wandb_project', type=str, default='out-test')
    logging_group.add_argument('--wandb_run_name', type=str, default='logs-test')

    ### Create model from json config file & save config file to json
    logging_group.add_argument('--load_config_json', type=str, help="Option to load model parameters from existing json file")
    logging_group.add_argument('--save_config_json', type=str, help="Option to save model parameters as new config json file")

    # Visualization args
    logging_group.add_argument('--statistic', choices=[ 'input_mean', 'input_median', 'input_stdev', 'input_max', 'input_min', 'output_mean', 'output_median', 'output_stdev', 'output_max', 'output_min', 'all_stats', 'input_all','output_all' ], default='input_mean', help='Select one or all statistics to display, e.g., --statistic input_min, or --statistic all_stats')
    logging_group.add_argument('--graph_type', choices=[ "heatmap", "plot", "boxplot", "all" ], default='no_graph', help='Select one of the graph types to display, e.g., --graph_type heatmap, or --graph_type plot')
    logging_group.add_argument('--box_plot_interval', default=1000, type=int, help='Instead of using mean/median/stdev statistics, create box plot of all input/output values at certain intervals of iteration')
    logging_group.add_argument('--box_plot_statistic', choices=['input', 'output', 'all'], default='', help='Select input or output statistic to display in boxplot')

    # Model Parameter Distribution
    logging_group.add_argument('--print_model_info', default=True, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if args.load_config_json is not None:
        with open(args.load_config_json, 'r') as config_file:
            config = json.load(config_file)

        # Update the args namespace with values from the JSON file
        for key, value in config.items():
            setattr(args, key, value)

    # Save all params to provided json if flag is present
    if args.save_config_json is not None:
        with open(args.save_config_json, 'w') as json_file:
            json.dump(vars(args), json_file)

    return args, model_group, training_group, logging_group

class Trainer:

    def __init__(self, args, model_group, training_group, logging_group):
        self.args = args
        self.model_group = model_group
        self.training_group = training_group
        self.logging_group = logging_group

        # typically make the decay iters equal to max_iters
        if self.args.lr_decay_match_max_iters:
            self.args.lr_decay_iters = self.args.max_iters

        self.setup()

        if self.args.sample_only:
            self.sample_and_print(self.args.max_sample_tokens, start_tokens=self.args.sample_start_tokens)

        if self.args.create_statistics:
            self.stats = initialize_statistics(self.args.n_layer, self.args.n_head)

    def setup(self):
        # Setup DDP
        self.ddp = int(os.environ.get('RANK', -1)) != -1
        if self.ddp:
            init_process_group(backend=self.args.backend)
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.ddp_local_rank}'
            print("this is my device", self.device)
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0
            self.seed_offset = self.ddp_rank
            self.args.gradient_accumulation_steps //= self.ddp_world_size
        else:
            self.device = self.args.device
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1

        self.tokens_per_iter = self.args.gradient_accumulation_steps * self.ddp_world_size * self.args.batch_size * self.args.block_size

        if self.master_process:
            os.makedirs(self.args.out_dir, exist_ok=True)

        print("seed: ", self.args.seed)
        print("seed offset: ", self.seed_offset)
        torch.manual_seed(self.args.seed + self.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.device_type = 'cuda' if 'cuda' in self.args.device else 'cpu'
        self.ptdtype = {"bfloat16" : torch.bfloat16, "float16" : torch.float16, "float32" : torch.float32}[self.args.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)

        # Model settings
        # TODO only add if they are defined from the argparse
        self.model_args = {action.dest: getattr(self.args, action.dest) for action in self.model_group._group_actions}
        self.model_args['vocab_size'] = None

        # Training settings
        self.training_args = {action.dest: getattr(self.args, action.dest) for action in self.training_group._group_actions}
        if self.args.dataset_list is not None:
            self.model_args['lsv_dataset_num'] = len(self.args.dataset_list)
            print("self.model_args['lsv_dataset_num']")
            print(self.model_args['lsv_dataset_num'])

        if self.args.init_from == 'scratch':
            self.model_args['vocab_size'] = self.get_vocab_size_from_meta()

            # Save full configuration used for training
            config_json = {**self.model_args, **self.training_args}
            with open(self.args.out_dir + "/full_config.json", "w") as configuration_file:
                json.dump(config_json, configuration_file, indent=4)
            with open(self.args.out_dir + "/best_val_loss_and_iter.txt", 'w') as file:
                print("resetting best val loss file")

            self.load_data()
            gptconf = GPTConfig(**self.model_args)
            self.model = GPT(gptconf)
            self.iter_num = 0 # for starting from scratch
            self.best_val_loss = 1e9 # really big number
        elif self.args.init_from == 'resume' or self.args.init_from == 'prev_run':
            if self.args.init_from == 'resume':
                ckpt_path = os.path.join(self.args.out_dir, 'ckpt.pt')
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                self.iter_num = checkpoint['iter_num']
            else:
                ckpt_path = os.path.join(self.args.prev_run_ckpt, 'ckpt.pt')
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                self.iter_num = 0

            # we should enforce that during resume training, the identical model args are used
            checkpoint_model_args = checkpoint['model_args']
            self.model_args = checkpoint_model_args

            # support for changing select params from resume (eg. for finetuning) based on cmd-line args entered (checks if diff than defaults)
            altered_model_args = {action.dest: getattr(self.args, action.dest) for action in self.model_group._group_actions if action.default != getattr(self.args, action.dest)}
            for k in altered_model_args:
                self.model_args[k] = altered_model_args[k]

            self.load_data()
            gptconf = GPTConfig(**self.model_args)
            self.model = GPT(gptconf)

            ## TODO: Add ability here to swap WTE factors.
            state_dict = checkpoint['model']
            for k,v in list(state_dict.items()):
                if k.startswith('_orig_mod.'):
                    state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
            self.model.load_state_dict(state_dict)
            self.best_val_loss = checkpoint['best_val_loss']
            if self.args.lsv_focused_training:
                self.model.freeze_non_lsv_parameters()

        elif self.args.init_from.startswith('gpt2'):

            assert self.args.gpt2_type in model_variation_dictionary

            self.iter_num = 0 # for starting from scratch
            self.best_val_loss = 1e9 # really big number

            variation_dict = model_variation_dictionary[self.args.gpt2_type]
            # NOTE: the hierarchy of parameters goes: 1)variation_dict >> 2)cmd-line args >> 3)GPTConfig defaults
            for k in variation_dict:
                self.model_args[k] = variation_dict[k]

            gptconf = GPTConfig(**self.model_args)
            self.model = GPT.from_pretrained(gptconf, model_type=self.args.gpt2_type)
            self.load_data()

            if self.args.lsv_focused_training:
                self.model.freeze_non_lsv_parameters()

        if self.args.block_size < self.model.config.block_size:
            self.model.crop_block_size(self.args.block_size)
            self.model_args['block_size'] = self.args.block_size

        self.model.to(self.device)

        # Print the model summary
        if self.args.print_model_info:
            print_summary(self.model)
            print_model_blocks(self.model)
            print_module_structure(self.model)
            print_model_tree(self.model, print_params=True)

        # Optimizer
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.args.dtype == 'float16'))
        self.optimizer = self.model.configure_optimizers(self.args.weight_decay, self.args.learning_rate,
                                                         (self.args.beta1, self.args.beta2), self.device_type)

        if self.args.compile:
            print("compiling the model... (takes a ~minute)")
            self.unoptimized_model = self.model
            self.model = torch.compile(self.model)

        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

        self.raw_model = self.model.module if self.ddp else self.model

        timestamp_prefix = time.strftime("%Y%m%d-%H%M%S")
        if self.args.timestamp:
            timestamp_prefix = self.args.timestamp

        # Tensorboard
        if self.args.tensorboard_log:
            timestamped_run_name = timestamp_prefix + "_" + self.args.tensorboard_run_name
            if self.args.csv_log:
                self.args.csv_name = timestamped_run_name
            log_subpath = os.path.join(self.args.tensorboard_log_dir, timestamped_run_name)
            self.writer = SummaryWriter(log_subpath)

        # Wandb
        if self.args.wandb_log and self.master_process:
            import wandb
            self.args.csv_name = wandb_run_name
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_run_name, config=self.args)
        self.load_tokenizer()

    def load_tokenizer(self):
        meta_path = os.path.join('data', self.args.dataset, 'meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            if 'tokenizer' in meta and meta['tokenizer'] == 'tiktoken':
                enc = tiktoken.get_encoding(meta['tiktoken_encoding'])
                print(f"Using tiktoken encoding: {meta['tiktoken_encoding']}")
                self.encode = lambda s: enc.encode(s, allowed_special={""})
                self.decode = lambda l: enc.decode(l)
            elif 'tokenizer' in meta and meta['tokenizer'] == 'sentencepiece':
                self.separator_token = "▁"
                self.stoi, self.itos = meta['stoi'], meta['itos']
                self.encode = lambda s: [self.stoi[c] for c in s]
                self.decode = lambda l: ''.join([self.itos[i] for i in l])
            else:
                self.stoi, self.itos = meta['stoi'], meta['itos']
                self.encode = lambda s: [self.stoi[c] for c in s]
                self.decode = lambda l: ''.join([self.itos[i] for i in l])
        else:
            raise FileNotFoundError(f"Meta file not found at {meta_path}")

    @torch.no_grad()
    def sample_and_print(self, max_sample_tokens, start_tokens="\n"):
        # Do one iteration per lsv, default to one with no lsv
        sample_iterations = 1

        if self.args.dataset_list is not None:
            sample_iterations = len(self.args.dataset_list)

        for i in range(sample_iterations):
            if self.args.use_lsv:
                self.model.set_lsv_index(i)
                print(f"lsv index {i}")

            start_ids = torch.tensor(self.encode(start_tokens), dtype=torch.long, device=self.device)[None, ...]
            x = start_ids

            with torch.no_grad():
                for _ in range(max_sample_tokens):
                    x_cond = x if x.size(1) <= self.args.block_size else x[:, -self.args.block_size:]
                    logits, _ = self.model(x_cond)
                    logits = logits[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                    x = torch.cat((x, next_id), dim=1)

            sampled_text = self.decode(x[0].tolist())
            print(f"Start tokens:\n{start_tokens}\n")
            print(f"Sampled text:\n{sampled_text}\n")

    def get_vocab_size_from_meta(self):
        # Data loader
        meta_path = os.path.join('data', self.args.dataset, 'meta.pkl')
        # Save a copy of meta.pkl tokenization into the output folder
        self.copy_file_to_directory(meta_path, self.args.out_dir)
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
                if 'vocab_size' in meta:
                    return meta['vocab_size']
                else:
                    sys.exit(f"Error: 'vocab_size' key not found in {meta_path}")
        else:
            sys.exit(f"Error: File not found - {meta_path}")

    def copy_file_to_directory(self, src_file, dest_dir):
        try:
            # Ensure the destination directory exists
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            # Copy the file
            shutil.copy(src_file, dest_dir)
            print(f"File {src_file} copied to {dest_dir}")
        except Exception as e:
            print(f"Error copying file: {e}")

    def load_data(self):

        if self.args.dataset_list is None:

            if self.model_args['vocab_size'] is None:
                sys.exit("Error: no vocab size specified")
            elif self.model_args['vocab_size'] == 100277:
                # cl100k_base, vocab size 100277, requires np.uint32
                self.train_data = np.memmap(os.path.join('data', self.args.dataset, 'train.bin'), dtype=np.uint32, mode='r')
                self.val_data = np.memmap(os.path.join('data', self.args.dataset, 'val.bin'), dtype=np.uint32, mode='r')
            else:
                # all other tokenations so far require only np.uint16
                self.train_data = np.memmap(os.path.join('data', self.args.dataset, 'train.bin'), dtype=np.uint16, mode='r')
                self.val_data = np.memmap(os.path.join('data', self.args.dataset, 'val.bin'), dtype=np.uint16, mode='r')
        else:
            self.train_data_dict = {}
            self.val_data_dict = {}

            for dataset in self.args.dataset_list:
                train_data = None
                val_data = None
                meta_path = os.path.join('data', dataset, 'meta.pkl')
                if not os.path.exists(meta_path):
                    sys.exit(f"Error: Meta file not found at {meta_path}")

                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                    vocab_size = meta.get('vocab_size', None)
                    if vocab_size:
                        self.model_args['vocab_size'] = vocab_size

                # Load train and val data for each dataset
                if self.model_args['vocab_size'] is None:
                    sys.exit("Error: no vocab size specified")
                elif self.model_args['vocab_size'] == 100277:
                    # cl100k_base, vocab size 100277, requires np.uint32
                    train_data = np.memmap(os.path.join('data', dataset, 'train.bin'), dtype=np.uint32, mode='r')
                    val_data = np.memmap(os.path.join('data', dataset, 'val.bin'), dtype=np.uint32, mode='r')
                else:
                    # all other tokenations so far require only np.uint16
                    train_data = np.memmap(os.path.join('data', dataset, 'train.bin'), dtype=np.uint16, mode='r')
                    val_data = np.memmap(os.path.join('data', dataset, 'val.bin'), dtype=np.uint16, mode='r')

                # Store in dictionaries
                self.train_data_dict[dataset] = train_data
                self.val_data_dict[dataset] = val_data


    def get_batch(self, split, target_dataset=None):
        dataset = None
        data = None
        def interpolate_probs(initial_probs, final_probs, method, step_ratio):
            if method == 'linear':
                return initial_probs + step_ratio * (final_probs - initial_probs)
            elif method == 'cosine':
                return initial_probs + 0.5 * (1 - np.cos(np.pi * step_ratio)) * (final_probs - initial_probs)
            elif method == 'exponential':
                return initial_probs * (final_probs / initial_probs) ** step_ratio
            else:
                raise ValueError(f"Unknown transition method: {method}")

        def get_transitioned_probs():
            initial_probs = np.array(self.args.dataset_sampling_probs)
            if self.args.final_dataset_sampling_probs:
                step_ratio = self.iter_num / self.args.max_iters
                final_probs = np.array(self.args.dataset_sampling_probs_final)
                return interpolate_probs(initial_probs, final_probs, self.args.transition_method, step_ratio)
            return initial_probs

        if self.args.dataset_list:
            # If multi-dataset sampling is enabled, pick a dataset using sampling probabilities
            if target_dataset:
                dataset = target_dataset
            elif self.args.dataset_interleaving:
                # print("using interleaving")
                if self.args.dataset_sampling_probs is not None:
                    # TODO: Move this section into README
                    # sampling proportions in this case
                    # allows for interleaving datasets
                    # Option 1) specific complex order
                    # a b a a b
                    # 1 1 1 1 1
                    # output: a b a a b
                    # Option 2) specific ratio shorthand
                    # a b c
                    # 1 3 2
                    # output: a b b b c c
                    # Option 3) specific ratio with random shuffle
                    # a b c
                    # 1 2 3
                    # possible random output: c a b c b c

                    # Init if does not exist
                    if not hasattr(self, 'remaining_datasets'):
                        self.remaining_datasets = []
                        # print("init")

                    # Reset if zero remaining
                    if len(self.remaining_datasets) == 0:
                        self.remaining_datasets = [x for x, count in zip(self.args.dataset_list, self.args.dataset_sampling_probs) for _ in range(int(count))]

                        # shuffle
                        if self.args.dataset_interleaving_shuffle:
                            random.shuffle(self.remaining_datasets)
                        # print("reset", self.remaining_datasets)

                    # pop from front of stack
                    dataset = self.remaining_datasets.pop(0)
                    # print("dataset", dataset, "remaining", self.remaining_datasets)
                else:
                    # If proportions and order not specified, then do 1:1 interleaving
                    num_datasets = len(self.args.dataset_list)
                    dataset_index = self.iter_num % num_datasets
                    dataset = self.args.dataset_list[dataset_index]

                data = self.train_data_dict[dataset] if split == 'train' else self.val_data_dict[dataset]
                # print(dataset)
            else:
                # print("using probabilities")
                if self.args.dataset_sampling_probs:
                    # Sample dataset based on probabilities
                    dataset = np.random.choice(self.args.dataset_list, p=get_transitioned_probs() / np.sum(get_transitioned_probs()))
                else:
                    # Default to uniform sampling if probabilities are not provided
                    dataset = np.random.choice(self.args.dataset_list)
                # print(dataset)

            if self.args.use_lsv:
                self.model.set_lsv_index(self.args.dataset_list.index(dataset))

            data = self.train_data_dict[dataset] if split == 'train' else self.val_data_dict[dataset]

            # set learning rate
            if self.args.dataset_sampling_learning_rate:
                dataset_index = self.args.dataset_list.index(dataset)
                self.args.learning_rate = self.args.dataset_sampling_learning_rate[dataset_index]

        else:
            # Else use the 'dataset' arg by default for backwards compatibility
            dataset = self.args.dataset
            data = self.train_data if split == 'train' else self.val_data

        # Generate random indices for the batch
        ix = torch.randint(len(data) - self.args.block_size, (self.args.batch_size,))

        # Get training and targets
        x = torch.stack([torch.from_numpy((data[i:i+self.args.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.args.block_size]).astype(np.int64)) for i in ix])

        # Send to appropriate device
        if self.device_type == 'cuda':
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    @torch.no_grad()
    def custom_loss_with_top1_focus(self, logits, targets):
        # Compute standard cross-entropy loss
        ce_loss = torch.nn.functional.cross_entropy(logits, targets)

        # Get the top-1 predictions
        top1_preds = torch.argmax(logits, dim=-1)

        # Focus more on the top-1 prediction by adding an additional term
        correct_top1 = (top1_preds == targets).float()  # 1 for correct, 0 for incorrect
        top1_focus_loss = 1.0 - correct_top1  # Emphasize the wrong top-1 predictions

        # Combine the original cross-entropy loss and the top-1 focus term
        loss = ce_loss + 0.5 * top1_focus_loss.mean()  # Adjust the weight (0.5) as needed
        return loss

    @torch.no_grad()
    def estimate_loss(self):
        out = {'datasets':{}}

        self.model.eval()
        # If multi-dataset sampling is enabled, we calculate loss per dataset
        if self.args.dataset_list and len(self.args.dataset_list) > 1:
            for dataset in self.args.dataset_list:
                print(f"Calculating loss for dataset: {dataset}")
                dataset_losses = {'train': torch.zeros(self.args.eval_iters), 'val': torch.zeros(self.args.eval_iters)}
                for split in ['train', 'val']:
                    for k in range(self.args.eval_iters):
                        X, Y = self.get_batch(split, target_dataset=dataset)
                        with self.ctx:
                            logits, loss = self.model(X, Y)
                        dataset_losses[split][k] = loss.item()
                out['datasets'][dataset] = {
                    'train': dataset_losses['train'].mean(),
                    'val': dataset_losses['val'].mean()
                }
            print("test")
            out['val'] = out['datasets'][self.args.dataset]['val']
            print(out['val'])

        else:
            # Default behavior for a single dataset
            for split in ['train', 'val']:
                losses = torch.zeros(self.args.eval_iters)
                for k in range(self.args.eval_iters):
                    X, Y = self.get_batch(split)
                    with self.ctx:
                        logits, loss = self.model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()

        self.model.train()
        return out

    def get_lr(self, it):
        if it < self.args.warmup_iters:
            return self.args.learning_rate * it / self.args.warmup_iters
        if it > self.args.lr_decay_iters:
            return self.args.min_lr
        decay_ratio = (it - self.args.warmup_iters) / (self.args.lr_decay_iters - self.args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.args.min_lr + coeff * (self.args.learning_rate - self.args.min_lr)

    def log_metrics(self, losses, lr, running_mfu, vram_allocated, iter_num, target_dataset=None):

        if self.args.tensorboard_log:
            # Log metrics for each dataset separately
            if target_dataset:
                self.writer.add_scalars(
                    "loss", {f"{target_dataset}/train": losses['train'].item(),
                             f"{target_dataset}/val": losses['val'].item()}, iter_num
                )
            else:
                self.writer.add_scalars(
                    "loss", {"train": losses['train'].item(), "val":
                             losses['val'].item()}, iter_num
                )

            self.writer.add_scalar("mfu_pct", running_mfu * 100, iter_num)
            self.writer.add_scalar("lr", lr, iter_num)
            self.writer.add_scalar("vram", vram_allocated, iter_num)

        if self.args.wandb_log and self.master_process:
            import wandb
            log_data = {
                "iter": iter_num,
                "lr": lr,
                "mfu": running_mfu * 100,
                "vram": vram_allocated,
            }
            if target_dataset:
                log_data[f"{dataset}/train/loss"] = losses['train']
                log_data[f"{dataset}/val/loss"] = losses['val']
            else:
                log_data["train/loss"] = losses['train']
                log_data["val/loss"] = losses['val']

            wandb.log(log_data)

        if self.args.csv_log:
            if target_dataset:
                self.write_to_csv(losses['train'].item(), losses['val'].item(), prefix=f"{target_dataset}_")
            else:
                self.write_to_csv(losses['train'].item(), losses['val'].item())

            # Other metrics
            self.write_to_csv(iter_num, lr, running_mfu, vram_allocated, prefix="misc_")




    def write_to_csv(self, *args, prefix=""):
        csv_full_dir = self.args.csv_dir
        if self.args.csv_ckpt_dir:
            csv_full_dir = f"{self.args.csv_dir}/{self.args.csv_ckpt_dir}"
        else:
            if self.args.tensorboard_log:
                csv_full_dir = f"{self.args.csv_dir}/{self.args.tensorboard_run_name.split('-')[0]}-{self.args.dataset}"
        os.makedirs(csv_full_dir, exist_ok=True)
        csv_path = os.path.join(csv_full_dir, prefix + self.args.csv_name + ".csv")
        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            # Write arguments as a new row in the CSV
            writer.writerow(args)


    def log_gamma_beta(self, gamma, beta, iter_num, layer_num, head_num=None):
        if self.args.tensorboard_log:
            if head_num:
                self.writer.add_scalars(
                        "gammas",
                        {"gamma_L" + str(layer_num) + "_H" + head_num: gamma},
                        iter_num
                        )
                self.writer.add_scalars(
                        "betas",
                        {"beta_L" + str(layer_num) + "_H" + head_num: beta},
                        iter_num
                        )
            else:
                self.writer.add_scalar( "gamma_L" + str(layer_num), gamma, iter_num)
                self.writer.add_scalar( "beta_L" + str(layer_num), beta, iter_num)

        if self.args.wandb_log and self.master_process:
            import wandb
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            })

    def log_metrics_non_validation(self, loss_training, running_mfu, vram_allocated, iter_num, target_dataset=None):
        if self.args.tensorboard_log:
            if target_dataset:
                self.writer.add_scalars(
                    "loss", {f"{target_dataset}/train": loss_training}, iter_num
                )
            else:
                self.writer.add_scalars(
                    "loss", { "train": loss_training }, iter_num
                )
            self.writer.add_scalar("mfu_pct", running_mfu * 100, iter_num)
            self.writer.add_scalar("vram", vram_allocated, iter_num)

        if self.args.wandb_log and self.master_process:
            import wandb
            wandb.log({
                "iter": iter_num,
                "train/loss": loss_training,
                "mfu": running_mfu*100,
                "vram": vram_allocated,
            })

    def train(self):
        self.X, self.Y = self.get_batch('train')
        t0 = time.time()
        local_iter_num = 0
        running_mfu = -1.0
        num_steps_with_worse_loss = 0
        graph_y_labels = []
        for layer in range(self.args.n_layer):
            for head in range(self.args.n_head):
                graph_y_labels.append(f"Layer {layer} Head {head}")

        # Create progress bar
        progress = Progress()
        with progress:
            task_id = progress.add_task("[green]Training...", total=(self.args.max_iters - self.iter_num))
            while True:
                lr = self.get_lr(self.iter_num) if self.args.decay_lr else self.args.learning_rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                if self.iter_num % self.args.eval_interval == 0 and self.master_process:
                    losses = self.estimate_loss()
                    vram_allocated = get_gpu_memory_info(info_type='used') if self.args.device != "cpu" else 0
                    if self.args.dataset_list is not None:
                        # Print loss for each dataset if multiple datasets are used
                        for dataset, dataset_losses in losses['datasets'].items():
                            print(f"step {self.iter_num}: {dataset} train loss {dataset_losses['train']:.4f}, val loss {dataset_losses['val']:.4f}")
                            self.log_metrics(dataset_losses, lr, running_mfu, vram_allocated, self.iter_num, target_dataset=dataset)
                    else:
                        # Default behavior for a single dataset
                        print(f"step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                        self.log_metrics(losses, lr, running_mfu, vram_allocated, self.iter_num)

                    # TODO: Support NaN checks for dataset_lists
                    if math.isnan(losses["val"]):
                        checkpoint = {
                            'model': self.raw_model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'model_args': self.model_args,
                            'iter_num': self.iter_num,
                            'best_val_loss': self.best_val_loss,
                            'nan_iter_num' : 0,
                            'nan' : True,
                            'config': vars(self.args),
                        }
                        torch.save(checkpoint, os.path.join(self.args.out_dir, 'ckpt.pt'))
                    if losses['val'] < self.best_val_loss or self.args.always_save_checkpoint:
                        if losses['val'] < self.best_val_loss:
                            self.iter_num_best_val_loss = self.iter_num
                            self.best_val_loss = losses['val']
                            # Save best validation loss
                            with open(os.path.join(self.args.out_dir, 'best_val_loss_and_iter.txt'), "w") as best_loss_file:
                                best_loss_file.write(str(self.best_val_loss.item())+","+str(self.iter_num))
                            # Reset early exit counter
                            num_steps_with_worse_loss = 0
                        if self.iter_num > 0:
                            checkpoint = {
                                'model': self.raw_model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'model_args': self.model_args,
                                'iter_num': self.iter_num,
                                'best_val_loss': self.best_val_loss,
                                'nan_iter_num' : None,
                                'nan' : None,
                                'config': vars(self.args),
                            }
                            print(f"saving checkpoint to {self.args.out_dir}")
                            # Save checkpoint
                            torch.save(checkpoint, os.path.join(self.args.out_dir, 'ckpt.pt'))
                        # Try new checkpoint if better val loss
                        if self.args.max_sample_tokens:
                            self.sample_and_print(self.args.max_sample_tokens, start_tokens=self.args.sample_start_tokens)
                        # export embedding table to npy file
                        if self.args.export_wte_npy:
                            self.raw_model.export_wte(self.args.export_wte_npy)
                        # export scale matrices to npz file
                        if self.args.export_scale_matrices_npz:
                            self.raw_model.export_scale_matrices(self.args.export_scale_matrices_npz)
                    else:
                        if self.args.sample_each_eval:
                            # Try model inference (e.g. exploring inference from overfitting)
                            if self.args.max_sample_tokens:
                                self.sample_and_print(self.args.max_sample_tokens, start_tokens=self.args.sample_start_tokens)
                        if self.args.export_wte_each_eval:
                            # export wte table to npy file
                            if self.args.export_wte_npy:
                                self.raw_model.export_wte(self.args.export_wte_npy)
                        if self.args.export_scale_matrices_each_eval:
                            # export scale matrices to npz file
                            if self.args.export_scale_matrices_npz:
                                self.raw_model.export_scale_matrices(self.args.export_scale_matrices_npz)

                    if self.args.patience is not None and num_steps_with_worse_loss >= self.args.patience:
                        print(f"Early Stopping: loss has not decreased in {self.args.patience + 1} steps")
                        break
                    if losses['val'] > self.best_val_loss:
                        num_steps_with_worse_loss += 1

                if self.args.eval_only:
                    break

                for micro_step in range(self.args.gradient_accumulation_steps):
                    if self.ddp:
                        self.model.require_backward_grad_sync = (micro_step == self.args.gradient_accumulation_steps - 1)

                    with self.ctx:
                        logits, loss = self.model(self.X, self.Y)

                        if self.args.focus_on_top1_loss:
                            loss = self.custom_loss_with_top1_focus(logits, self.Y)  # Use custom loss

                        loss = loss / self.args.gradient_accumulation_steps

                    self.X, self.Y = self.get_batch('train')

                    self.scaler.scale(loss).backward()

                if self.args.grad_clip != 0.0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.optimizer.zero_grad(set_to_none=True)

                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                if self.iter_num % self.args.log_interval == 0 and self.master_process:
                    lossf = loss.item() * self.args.gradient_accumulation_steps
                    if local_iter_num >= 5:
                        mfu = self.raw_model.estimate_mfu(self.args.batch_size * self.args.gradient_accumulation_steps, dt)
                        running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                    print(f"iter {self.iter_num}: loss {lossf:.4f}, time {dt*1000:.2f} ms, mfu {running_mfu*100:.2f}%")
                    if math.isnan(lossf):
                        if self.args.save_nan_checkpoint:
                            checkpoint = {
                                'model': self.raw_model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'model_args': self.model_args,
                                'iter_num': self.iter_num_best_val_loss,
                                'best_val_loss': self.best_val_loss,
                                'nan_iter_num' : self.iter_num,
                                'nan' : True,
                                'config': vars(self.args),
                            }
                            print(f"saving checkpoint to {self.args.out_dir}")
                            torch.save(checkpoint, os.path.join(self.args.out_dir, 'ckpt.pt'))
                        sys.exit("Exiting training loss is NaN")

                    vram_allocated = get_gpu_memory_info(info_type='used') if self.args.device != "cpu" else 0
                    self.log_metrics_non_validation(lossf, running_mfu, vram_allocated, self.iter_num)


                if self.args.create_statistics and local_iter_num % self.args.softmax_io_log_interval == 0:
                    create_statistics(self, graph_y_labels)


                self.iter_num += 1
                local_iter_num += 1

                # Update progress bar
                progress.update(task_id, advance=1)

                # End of training actions
                if self.iter_num > self.args.max_iters:
                    if self.args.only_save_checkpoint_at_end:
                        checkpoint = {
                            'model': self.raw_model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'model_args': self.model_args,
                            'iter_num': self.iter_num,
                            'best_val_loss': self.best_val_loss,
                            'nan_iter_num' : None,
                            'nan' : None,
                            'config': vars(self.args),
                        }
                        print(f"saving checkpoint to {self.args.out_dir}")
                        torch.save(checkpoint, os.path.join(self.args.out_dir, 'ckpt.pt'))
                        # Sample if set
                        if self.args.max_sample_tokens:
                            self.sample_and_print(self.args.max_sample_tokens, start_tokens=self.args.sample_start_tokens)
                    break

            if self.args.plot_statistics:
                plot_statistics(self.args, self.stats, graph_y_labels)

            if self.args.tensorboard_log:
                self.writer.flush()
                self.writer.close()

            if self.args.wandb_log and self.master_process:
                import wandb
                wandb.log({"finished": True})
                wandb.finish()

def main():
    args, model_group, training_group, logging_group = parse_args()
    trainer = Trainer(args, model_group, training_group, logging_group)

    if not args.sample_only:
        trainer.train()

    if trainer.ddp:
        destroy_process_group()

    if args.tensorboard_log:
        trainer.writer.flush()
        trainer.writer.close()

if __name__ == '__main__':
    main()


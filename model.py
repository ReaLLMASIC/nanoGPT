# model.py
"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
import sys
from rich import print
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

# Checkpointing
import torch.utils.checkpoint as checkpoint

from gpt_conf import GPTConfig

# Variations
from variations.attention_variations import attention_dictionary
from variations.mlp_variations import get_mlp_instance
from variations.moe_variations import MoELayer
from variations.lsv_variations import lsv_dictionary
from variations.softmax_variations import softmax_dictionary
from variations.norm_variations import norm_dictionary
from variations.position_encoding_variations import QuantizedEmbedding, RotaryEmbedding, SymmetricalOverlapAngularPositions, FIRE
from variations.activation_variations import activation_dictionary
from variations.linear_variations import linear_dictionary
from variations.router_variations import router_dictionary
from quantization.quantize import quantize_dictionary, dequantize, fake_quantize_act
from quantization.quant_utils import set_variant, create_activation_buffers

def create_shared_param_group(layer_type, config):
    """
    Creates a shared list of layer blocks (either MLP or Attn), optionally reusing blocks
    every 'shared_size' layers, and optionally reflecting them symmetrically if
    'shared_sym' is True. Also can handle multiple attention variants in one model.

    Args:
        layer_type (str): "mlp" or "attn"
        config: a config object containing fields like:
            - n_layer (int): number of layers total
            - use_moe (bool): if True, some MLP layers replaced by MoE
            - moe_layer_freq (int): frequency of MoE layers
            - shared_mlp_size, shared_attn_size (int)
            - shared_mlp_sym, shared_attn_sym (bool)
            - shared_fire_embeddings (bool)
            - attention_variants (list of str): e.g. ["causal", "fancy"] ...
              if you want multiple attention types
            - n_head, ...
        Returns:
            list of layer_blocks
    """

    # If you use FIRE for position embeddings in the attention layer:
    from variations.position_encoding_variations import FIRE

    # Determine if we are building MLP or attn blocks
    if layer_type == "mlp":
        shared_size = config.shared_mlp_size
        shared_sym = config.shared_mlp_sym
    elif layer_type == "attn":
        shared_size = config.shared_attn_size
        shared_sym = config.shared_attn_sym
    else:
        sys.exit(f"{layer_type} not supported, exiting")

    # If attn layer, optionally build a single FIRE module to share
    fire_pos_enc = None
    if layer_type == "attn" and config.shared_fire_embeddings:
        fire_pos_enc = FIRE(config, num_heads=config.n_head)

    shared_group = []
    layer_block = None

    for i in range(config.n_layer):

        # Create a new layer block every "shared_size"
        if i % shared_size == 0:
            if layer_type == "mlp":
                # Possibly handle MoE
                if config.use_moe and i % config.moe_layer_freq == 0:
                    layer_block = MoELayer(config)
                else:
                    layer_block = get_mlp_instance(config)

            elif layer_type == "attn":
                attn_cls = attention_dictionary[config.attention_variant]

                # Instantiate an attention layer
                layer_block = attn_cls(config, fire_pos_enc=fire_pos_enc)
            else:
                sys.exit(f"{layer_type} not supported, exiting")

        # Add this (possibly reused) block to the list
        shared_group.append(layer_block)

        # If symmetrical sharing is requested
        if shared_sym:
            # Even number of layers
            if config.n_layer % 2 == 0:
                # Once we reach halfway-1, we append the blocks in reverse
                if i == (config.n_layer // 2 - 1):
                    for j in range(i + 1):
                        shared_group.append(shared_group[i - j])
                    return shared_group
            else:
                # Odd number of layers
                if i == (config.n_layer // 2):
                    for j in range(i):
                        shared_group.append(shared_group[i - j])
                    return shared_group

    return shared_group


class Block(nn.Module):
    def __init__(self, config, mlp=None, attn=None):
        super().__init__()

        # Initialize and set attn normalization (e.g. rmsnorm)
        norm_variant_attn = norm_dictionary[config.norm_variant_attn]
        self.ln_1 = norm_variant_attn(config)
        if not config.use_parallel_mlp:
            self.ln_2 = norm_variant_attn(config)

        self.use_post_ln = config.use_post_ln
        self.use_parallel_mlp = config.use_parallel_mlp
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

        # Allow for sharing attn between blocks
        if attn is None:
            self.attn = attention_dictionary[config.attention_variant](config)
        else:
            self.attn = attn

        # Allow for sharing mlp between blocks
        if mlp is None:
            self.mlp = get_mlp_instance(config)
        else:
            self.mlp = mlp

    def forward(self, x, iter_num):
        def custom_forward(*inputs):
            x = inputs[0]
            if self.use_post_ln:
                if self.use_parallel_mlp:
                    x = self.ln_1(x + self.attn(x, iter_num) + self.mlp(x, iter_num))
                else:
                    x = self.ln_1(x + self.attn(x, iter_num))
                    x = self.ln_2(x + self.mlp(x, iter_num))
            else:
                if self.use_parallel_mlp:
                    ln_1 = self.ln_1(x)
                    x = x + self.attn(ln_1, iter_num) + self.mlp(ln_1, iter_num)
                else:
                    x = x + self.attn(self.ln_1(x), iter_num)
                    x = x + self.mlp(self.ln_2(x), iter_num)
            return x

        if self.use_gradient_checkpointing and x.requires_grad:
            return checkpoint.checkpoint(custom_forward, x, use_reentrant=False)
        else:
            return custom_forward(x)

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config

        # Shared Parameters MLP
        shared_mlp_array = create_shared_param_group("mlp", config)
        # Shared Parameters Attention
        shared_attn_array = create_shared_param_group("attn", config)

        # Factorization Parameters
        self.n_embd_wte = config.n_embd_wte
        self.n_embd_wte_scale_tying = config.n_embd_wte_scale_tying

        # Learned Steering Vectors
        self.use_lsv = config.use_lsv
        self.lsv_index = config.lsv_index
        self.lsv_dataset_num = config.lsv_dataset_num

        if config.lsv_dataset_num is not None and config.use_lsv:
            self.num_datasets = config.lsv_dataset_num
            print(config.lsv_variant)
            self.lsv_variant = config.lsv_variant
            self.lsv_matrix = lsv_dictionary[self.lsv_variant](config)

        self.transformer = nn.ModuleDict(dict())
        # Configure wte, with optional quantization and factoring
        if config.quantize_wte:
            if config.n_embd_wte:
                # If factorization is set
                word_embd = QuantizedEmbedding(config.vocab_size, config.n_embd_wte, config.quantize_wte_method, config.quantize_wte_bits)
            else:
                # no factorization
                word_embd = QuantizedEmbedding(config.vocab_size, config.n_embd, config.quantize_wte_method, config.quantize_wte_bits)
            self.transformer['wte'] = word_embd
        else:
            if config.n_embd_wte:
                # If factorization is set
                word_embd = nn.Embedding(config.vocab_size, config.n_embd_wte)
                self.transformer['wte'] = word_embd
            else:
                #TODO: currently multicontext is in own category, add support later for WTE factorization
                if config.multicontext:
                    for i, vocab_size in enumerate(self.config.vocab_sizes):
                        embedding_layer = nn.Embedding(vocab_size, config.n_embd)
                        self.transformer[f'wte_{i}'] = embedding_layer
                        self.transformer[f'lm_head_{i}'] = nn.Linear(config.n_embd, vocab_size, bias=False)
                        self.transformer[f'lm_head_{i}'].weight = self.transformer[f'wte_{i}'].weight
                else:
                    # no factorization
                    word_embd = nn.Embedding(config.vocab_size, config.n_embd)
                    self.transformer['wte'] = word_embd


        self.transformer['drop'] = nn.Dropout(config.dropout)
        self.transformer['h'] = nn.ModuleList([Block(config, mlp=shared_mlp_array[i], attn=shared_attn_array[i]) for i in range(config.n_layer)])
        self.transformer['ln_f'] = norm_dictionary[config.norm_variant_output](config)

        if self.config.use_abs_pos_embeddings:
            if config.quantize_wpe:
                pos_embd = QuantizedEmbedding(config.block_size, config.n_embd, config.quantize_wpe_method, config.quantize_wpe_bits)
            else:
                pos_embd = nn.Embedding(config.block_size, config.n_embd)
            self.transformer['wpe'] = pos_embd

        # Select softmax variant for output layer
        self.softmax_variant_output = config.softmax_variant_output
        if self.softmax_variant_output != "softmax":
            self.softmax_layer_output = softmax_dictionary[config.softmax_variant_output](config)

        if config.n_embd_wte:
            self.lm_head = nn.Linear(config.n_embd_wte, config.vocab_size, bias=False)
            self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        else:
            #TODO: currently multicontext is in own category, add support later for WTE factorization
            if config.multicontext:
                for i, vocab_size in enumerate(self.config.vocab_sizes):
                    self.transformer[f'lm_head_{i}'].weight = self.transformer[f'wte_{i}'].weight
            else:
                self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
                # with weight tying when using torch.compile() some warnings get generated:
                # "UserWarning: functional_call was passed multiple values for tied weights.
                # This behavior is deprecated and will be an error in future versions"
                # not 100% sure what this is, so far seems to be harmless. TODO investigate
                self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # Initialize and possibly import scale_up and scale_down matrices, if factorization is set
        if self.n_embd_wte:
            # TODO: make this linear set from variant dictionary
            # TODO: make this linear quantizable
            self.transformer['scale_up'] = nn.Linear(config.n_embd_wte, config.n_embd, bias=False)
            self.transformer['scale_down'] = nn.Linear(config.n_embd_wte, config.n_embd, bias=False)

            if self.n_embd_wte_scale_tying:
                self.transformer.scale_up.weight = self.transformer.scale_down.weight # Weight tying

            if config.import_scale_matrices_freeze:
                self.transformer.scale_up.weight.requires_grad = False
                self.transformer.scale_down.weight.requires_grad = False

        # init all weights
        self.apply(self._init_weights)

        # import wte
        if self.config.import_wte_npy:
            # Replace wte with values from numpy and retie weights
            self.import_wte(self.config.import_wte_npy)

        # import scale_matrices
        if config.import_scale_matrices_npz:
            self.import_scale_matrices(config.import_scale_matrices_npz, config.n_embd_wte_scale_tying)

        for pn, p in self.named_parameters():
            # apply special scaled init to the residual projections, per GPT-2 paper
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.config.use_abs_pos_embeddings:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def update_block_size(self, new_block_size):
        # Function to increase block size dynamically
        if new_block_size > self.config.block_size:
            self.config.block_size = new_block_size
            if self.config.use_abs_pos_embeddings:
                if self.config.quantize_wpe:
                    pos_embd = QuantizedEmbedding(new_block_size, self.config.n_embd, self.config.quantize_wpe_method, self.config.quantize_wpe_bits)
                else:
                    pos_embd = nn.Embedding(new_block_size, self.config.n_embd)
                self.transformer.wpe = pos_embd
            for block in self.transformer.h:
                if hasattr(block.attn, 'bias'):
                    block.attn.bias = torch.tril(torch.ones(new_block_size, new_block_size)).view(1, 1, new_block_size, new_block_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=self.config.linear_mean_init, std=self.config.linear_std_init)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=self.config.embedding_mean_init, std=self.config.embedding_std_init)

    def update_num_angles(self, num_angles):
        """Update the number of angles for rotary embeddings in all attention layers."""
        device = next(self.parameters()).device
        for block in self.transformer.h:
            if hasattr(block.attn, 'rotary_emb_q') and hasattr(block.attn, 'rotary_emb_k'):
                block.attn.rotary_emb_q.update_num_angles(num_angles, device)
                block.attn.rotary_emb_k.update_num_angles(num_angles, device)

    def update_rope_length(self, rope_length):
        """Update the number of angles for rotary embeddings in all attention layers."""
        for block in self.transformer.h:
            if hasattr(block.attn, 'rotary_emb_q') and hasattr(block.attn, 'rotary_emb_k'):
                block.attn.rotary_emb_q.update_rope_length(rope_length)
                block.attn.rotary_emb_k.update_rope_length(rope_length)

    def import_wte(self, file_path):
        """ Replace wte with values from numpy and retie weights """

        #Load and format weights
        initial_embeddings = np.load(self.config.import_wte_npy)
        initial_embeddings_tensor = torch.from_numpy(initial_embeddings).float()

        # Initialize imported wte
        self.transformer.wte = nn.Embedding.from_pretrained(
                initial_embeddings_tensor,
                freeze=self.config.import_wte_freeze
                )

        # Redo the Weight tying
        self.lm_head.weight = self.transformer.wte.weight

    def export_wte(self, file_path):
        # TODO: Determine strategy with this and other means of export, possibly
        # replacing this with composition of existing means
        embedding_table = self.transformer.wte.weight.detach().cpu().numpy()
        np.save(file_path, embedding_table)
        print(f"Embedding table saved to {file_path}")

    def import_scale_matrices(self, file_path, weight_tying=False):
        """Import scale_up and scale_down matrices from a numpy file."""
        scale_matrices = np.load(file_path)
        scale_up_tensor = torch.from_numpy(scale_matrices['scale_up']).float().T
        scale_down_tensor = torch.from_numpy(scale_matrices['scale_down']).float().T

        print(scale_up_tensor.size())
        print(scale_down_tensor.size())
        self.transformer.scale_up.weight.data.copy_(scale_up_tensor)
        self.transformer.scale_down.weight.data.copy_(scale_down_tensor)

        if weight_tying:
            self.transformer.scale_up.weight = self.transformer.scale_down.weight

        print(f"Scale matrices loaded from {file_path} with weight tying: {weight_tying}")

    def export_scale_matrices(self, file_path):
        """Export scale_up and scale_down matrices to a numpy file."""
        scale_up_matrix = self.transformer.scale_up.weight.detach().cpu().numpy()
        scale_down_matrix = self.transformer.scale_down.weight.detach().cpu().numpy()

        np.savez(file_path, scale_up=scale_up_matrix, scale_down=scale_down_matrix)
        print(f"Scale matrices saved to {file_path}")

    def forward(self, idx, targets=None, token_dict=None, target_dict=None, iter_num=None):
        if token_dict is not None:

            token_list = list(token_dict.values())
            target_list = list(target_dict.values())
            device = token_list[0].device
            b, t = token_list[0].size()

            # Add all of the input tokens
            for i, tokens in enumerate(token_list):
                if i == 0:
                    x = self.transformer[f'wte_{i}'](tokens)
                else:
                    x += self.transformer[f'wte_{i}'](tokens)

            if self.config.use_abs_pos_embeddings:
                pos = torch.arange(0, t, dtype=torch.long, device=device)
                pos_emb = self.transformer.wpe(pos)  # (t, n_embd)
                x = self.transformer.drop(x + pos_emb)
            else:
                x = self.transformer.drop(x)

            x.requires_grad_(True)

            # 2. Possibly apply LSV on input
            if self.use_lsv and self.config.apply_lsv_at_layer_idx == 0:
                x = self.lsv_matrix(x)

            layer_idx = 1
            for block in self.transformer.h:
                if self.config.use_gradient_checkpointing:
                    x = checkpoint.checkpoint(block, x, iter_num, use_reentrant=self.config.recompute_backward_pass)
                else:
                    x = block(x, iter_num)

                # Steering logic
                if self.use_lsv and layer_idx == self.config.apply_lsv_at_layer_idx:
                    x = self.lsv_matrix(x)
                if (self.config.apply_vector_at_layer_idx is not None
                        and layer_idx == self.config.apply_vector_at_layer_idx):
                    x = self.apply_vector_to_layer_output(x)
                if (self.config.obtain_vector_at_layer_idx is not None
                        and layer_idx == self.config.obtain_vector_at_layer_idx):
                    x = self.obtain_vector_from_layer_output(x)

                layer_idx += 1

            # 3. Final layer norm
            x = self.transformer.ln_f(x)

            # 4. Optionally scale down
            if self.n_embd_wte:
                x = F.linear(x, self.transformer.scale_down.weight.t())

            # 5. Compute separate logits
            logits = []
            for i in range(len(token_list)):
                logits.append(self.transformer[f'lm_head_{i}'](x))

            # 6. Compute losses if targets are provided
            # If we only want the last token, adapt the slices as you prefer
            losses = []
            if target_list is not None:
                for i in range(len(token_list)):
                    losses.append(
                            F.cross_entropy(
                                logits[i].view(-1, logits[i].size(-1)),
                                target_list[i].view(-1),
                                ignore_index=-1
                                )
                            )

            else:
                # only forward lm head on very last position in inference mode
                for i in range(len(token_list)):
                    logits.append(self.transformer[f'lm_head_{i}'](x[:, [-1], :]))
                losses = None

            return logits, losses

        else:
            device = idx.device
            b, t = idx.size()
            # assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

            # forward the GPT model itself
            tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
            x = None

            if self.n_embd_wte:
                tok_emb = self.transformer.scale_up(tok_emb)
            if self.config.use_abs_pos_embeddings:
                pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
                pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
                x = self.transformer.drop(tok_emb + pos_emb)
            else:
                x = self.transformer.drop(tok_emb)

            x.requires_grad_(True)  # Ensure requires_grad is True

            if self.use_lsv and self.config.apply_lsv_at_layer_idx == 0:
                x = self.lsv_matrix(x)

            layer = 1
            for block in self.transformer.h:
                # Propagate tokens through layers
                if self.config.use_gradient_checkpointing:
                    x = checkpoint.checkpoint(block, x, iter_num, use_reentrant=self.config.recompute_backward_pass)
                else:
                    x = block(x, iter_num)

                # Intercept for Learned Steering Vectors
                if self.use_lsv and layer == self.config.apply_lsv_at_layer_idx:
                    x = self.lsv_matrix(x)
                    # x = self.apply_learned_vector_to_layer_output(x)

                # Intercept for Steering Vectors
                if self.config.apply_vector_at_layer_idx is not None and layer == self.config.apply_vector_at_layer_idx:
                    x = self.apply_vector_to_layer_output(x)
                if self.config.obtain_vector_at_layer_idx is not None and layer == self.config.obtain_vector_at_layer_idx:
                    print(layer, self.config.obtain_vector_at_layer_idx)
                    x = self.obtain_vector_from_layer_output(x)

                layer +=1

            x = self.transformer.ln_f(x)

            if self.n_embd_wte:
                x = F.linear(x, self.transformer.scale_down.weight.t())

            if targets is not None:
                # if we are given some desired targets also calculate the loss
                logits = self.lm_head(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            else:
                # inference-time mini-optimization: only forward the lm_head on the very last position
                logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
                loss = None

            return logits, loss

    def set_lsv_scaling_factor(self, factor):
        self.lsv_matrix.update_lsv_scaling_factor(factor)

    def set_lsv_mode(self, mode):
        self.lsv_matrix.set_mode(mode)

    def set_lsv_mixture(self, mixture):
        """ Mixture is a list, allowing for mixing steering vectors """
        self.lsv_matrix.set_mixture(mixture)

    def get_lsv_scaling_factor(self):
        return self.lsv_matrix.get_lsv_scaling_factor()

    def set_lsv_index(self, index):
        self.lsv_matrix.update_lsv_index(index)

    def freeze_non_lsv_parameters(self):
        """Freeze all parameters except for lsv_matrix if lsv_focused_training is enabled."""

        print("Freezing all parameters except for lsv_matrix")

        # Freeze all parameters by setting requires_grad to False
        for name, param in self.named_parameters():
            if name != "lsv_matrix":
                param.requires_grad = False
            else:
                param.requires_grad = True  # Ensure lsv_matrix can still be trained

    def apply_learned_vector_to_layer_output(self, x):
        """Conditionally add a vector based on dataset index to the output of a specific layer."""

        # Use one-hot vector for the dataset and multiply by the learned parameter matrix
        one_hot_vector = torch.zeros(self.lsv_matrix.size(0), device=x.device)
        one_hot_vector[self.lsv_index] = 1.0

        # Multiply the one-hot vector by the learned parameter matrix
        selected_vector = torch.matmul(one_hot_vector, self.lsv_matrix)

        x = x + selected_vector

        return x

    def apply_vector_to_layer_output(self, x):
        """Conditionally add a vector from a file to the output of a specific layer."""

        # require this method has the vector file
        assert self.config.apply_vector_file is not None

        vector = np.load(self.config.apply_vector_file)
        vector_tensor = torch.from_numpy(vector).float().to(x.device)
        x = x + self.config.apply_vector_scaling_factor * vector_tensor

        return x

    def obtain_vector_from_layer_output(self, x):
        """Append a vector to an existing .npy file."""

        # Convert the tensor back to a numpy array
        y = x
        y = torch.mean(y, dim=1, keepdim=True)
        result_vector = y.detach().cpu().numpy()

        # Save the vector to file
        np.save(self.config.obtain_vector_file, result_vector)
        print(f"Updated avg vector saved to {self.config.obtain_vector_file}")

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if self.config.use_abs_pos_embeddings:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, config, model_type):
        # assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        print(f"loading weights from pretrained gpt: {model_type}")

        # create a from-scratch initialized minGPT model
        model = GPT(config)
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        # NOTE: the assert below will fail because we split out the c_attn linears!
        # assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for key in sd_keys_hf:
            if any(key.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[key].shape[::-1] == sd[key].shape
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key].t())
            elif key.endswith('attn.c_attn.weight') or key.endswith('attn.c_attn.bias'):
                # split into c_attn_q/k/v
                q, k, v  = sd_hf[key].t().split(config.n_embd, dim=0)
                q_key_str = key.replace("c_attn", "c_attn_q")
                k_key_str = key.replace("c_attn", "c_attn_k")
                v_key_str = key.replace("c_attn", "c_attn_v")
                sd[q_key_str] = q
                sd[k_key_str] = k
                sd[v_key_str] = v
            else:
                # vanilla copy over the other parameters
                print(key)
                if config.n_embd_wte:
                    if key == "transformer.wte.weight":
                        continue
                    if key == "lm_head.weight":
                        continue

                if not config.use_abs_pos_embeddings:
                    if key == "transformer.wpe.weight":
                        continue

                assert sd_hf[key].shape == sd[key].shape
                with torch.no_grad():
                    print(key)
                    sd[key].copy_(sd_hf[key])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = None
            if self.config.softmax_variant_output != 'softmax':
                probs = self.softmax_layer_output(logits)
            else:
                probs = F.softmax(logits, dim=-1)
            assert probs != None
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def generate_with_stop(self, idx, max_new_tokens, stop_string, decode, temperature=1.0, top_k=None):
        """
        Generate tokens and stop on fixed string match, return the state for further input.
        """
        generated_text = ""
        buffer = ""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            next_token_text = decode(idx_next[0].tolist())
            generated_text += next_token_text
            buffer += next_token_text

            # Check if the buffer ends with the stop_string
            if buffer.endswith(stop_string):
                break

        return idx, generated_text


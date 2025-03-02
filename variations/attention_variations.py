# variations/attention_variations.py

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from variations.linear_variations import linear_dictionary
from quantization.quantize import fake_quantize_act
from quantization.quant_utils import set_variant, create_activation_buffers
from variations.softmax_variations import softmax_dictionary
from variations.position_encoding_variations import QuantizedEmbedding, RotaryEmbedding, SymmetricalOverlapAngularPositions, FIRE

# Mamba related imports
if torch.cuda.is_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

class CausalSelfAttention(nn.Module):
    def __init__(self, config, fire_pos_enc=None):
        super().__init__()

        self.full_quant_iteration = config.full_quant_iteration
        self.eval_interval = config.eval_interval
        self.start_quant_level = config.start_quant_level
        self.quant_scheduler = config.quant_scheduler

        if (config.n_kv_group == None):
            config.n_kv_group = config.n_head
        else:
            assert config.n_embd % config.n_kv_group == 0

        self.quantization_attn_dict = {}
        self.quantization_attn_dict["activations_quant_method"] = config.activations_quant_method
        for arg, val in vars(config).items():
            # Set each attention Activation precision and method
            if arg.startswith("quantize_") and "attn_act" in arg and arg.endswith("_bits"):
                self.quantization_attn_dict[arg] = set_variant(val, config.quantize_attn_act_bits)
            elif arg.startswith("quantize_") and "attn_act" in arg:
                self.quantization_attn_dict[arg] = set_variant(val, config.quantize_attn_act)
                if config.store_activations and arg != "quantize_attn_act" and self.quantization_attn_dict[arg]:
                    create_activation_buffers(self, arg)
            # Set each attention Linear precision and method
            elif arg.startswith("quantize_") and "linear_attn" in arg and arg.endswith("_bits"):
                self.quantization_attn_dict[arg] = set_variant(val, config.quantize_linear_bits)
            elif arg.startswith("quantize_") and "linear_attn" in arg and arg.endswith("_method"):
                self.quantization_attn_dict[arg] = set_variant(val, config.quantize_linear_method)

        self.linear_variant_q = linear_dictionary[set_variant(config.linear_variant_q, config.linear_variant_attn)]
        self.linear_variant_k = linear_dictionary[set_variant(config.linear_variant_k, config.linear_variant_attn)]
        self.linear_variant_v = linear_dictionary[set_variant(config.linear_variant_v, config.linear_variant_attn)]
        self.linear_variant_attn_proj = linear_dictionary[set_variant(config.linear_variant_attn_proj, config.linear_variant_attn)]

        # key, query, value projections for all heads, but in a batch
        self.c_attn_q = self.linear_variant_q(config.n_embd, config.n_embd, config, self.quantization_attn_dict["quantize_linear_attn_q_method"], self.quantization_attn_dict["quantize_linear_attn_q_bits"], bias=config.bias)

        self.n_head = config.n_head
        if config.n_kv_group == None:
            self.n_kv_group = config.n_head
        else:
            assert config.n_head % config.n_kv_group == 0
            self.n_kv_group = config.n_kv_group

        self.kv_dim = (config.n_embd // config.n_head) * self.n_kv_group
        self.c_attn_k = self.linear_variant_k(config.n_embd, self.kv_dim, config, self.quantization_attn_dict["quantize_linear_attn_k_method"], self.quantization_attn_dict["quantize_linear_attn_k_bits"], bias=config.bias)
        self.c_attn_v = self.linear_variant_v(config.n_embd, self.kv_dim, config, self.quantization_attn_dict["quantize_linear_attn_v_method"], self.quantization_attn_dict["quantize_linear_attn_v_bits"], bias=config.bias)
        self.c_proj = self.linear_variant_attn_proj(config.n_embd, config.n_embd, config, self.quantization_attn_dict["quantize_linear_attn_proj_method"], self.quantization_attn_dict["quantize_linear_attn_proj_bits"], bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # Embedding
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.n_embd = config.n_embd
        self.gate = config.gate
        self.use_fire_embeddings = None
        self.disable_flash_attention = config.disable_flash_attention
        if config.use_fire_embeddings:
            self.use_fire_embeddings = config.use_fire_embeddings
            if fire_pos_enc is not None:
                self.fire_pos_enc = fire_pos_enc
                print("shared fire")
            else:
                self.fire_pos_enc = FIRE(config, num_heads=config.n_head)
                print("indiv fire")

        # Rotary Positional Embeddings
        self.rotary_emb_q = None
        self.rotary_emb_k = None
        if config.use_rotary_embeddings:
            # Note: size is the size of the head dimension
            if config.rope_variant == "soap":
                self.sym_rot_num_angles = config.sym_rot_num_angles
                self.rotary_emb_q = SymmetricalOverlapAngularPositions(config, size=config.n_embd // self.n_head, num_angles=self.sym_rot_num_angles)
                self.rotary_emb_k = SymmetricalOverlapAngularPositions(config, size=config.n_embd // self.n_head, num_angles=self.sym_rot_num_angles)
            elif config.rope_variant == "rope":
                self.rotary_emb_q = RotaryEmbedding(config, size=config.n_embd // self.n_head)
                self.rotary_emb_k = RotaryEmbedding(config, size=config.n_embd // self.n_head)

        # Sliding window size
        self.window_size = config.window_size
        print(f"sliding window size: {self.window_size}")

        # Using flex attention
        self.use_flex_attn = config.use_flex_attn

        # Gating
        self.gate = config.gate

        # Fire Embeddings
        self.use_fire_embeddings = None
        if config.use_fire_embeddings:
            self.use_fire_embeddings = config.use_fire_embeddings
            if fire_pos_enc is not None:
                self.fire_pos_enc = fire_pos_enc
                print("shared fire")
            else:
                self.fire_pos_enc = FIRE(config, num_heads=config.n_head)
                print("indiv fire")

        # Rotary Positional Embeddings
        self.rotary_emb_q = None
        self.rotary_emb_k = None
        if config.use_rotary_embeddings:
            # Note: size is the size of the head dimension
            if config.rope_variant == "soap":
                self.sym_rot_num_angles = config.sym_rot_num_angles
                self.rotary_emb_q = SymmetricalOverlapAngularPositions(config, size=config.n_embd // self.n_head, num_angles=self.sym_rot_num_angles)
                self.rotary_emb_k = SymmetricalOverlapAngularPositions(config, size=config.n_embd // self.n_head, num_angles=self.sym_rot_num_angles)
            elif config.rope_variant == "rope":
                self.rotary_emb_q = RotaryEmbedding(config, size=config.n_embd // self.n_head)
                self.rotary_emb_k = RotaryEmbedding(config, size=config.n_embd // self.n_head)


        self.flash = True
        if self.window_size is not None:
            # TODO: look into supporting sliding window attn for flash attn
            self.flash = False
            print("flash attention removed due to windowed attention")

        if self.n_kv_group != self.n_head:
            self.flash = False
            print("flash attention removed due to GQA")

        if self.use_fire_embeddings:
            self.flash = False
            print("flash attention removed due to FIRE")

        # Can't use flash attention if we want to manually quantize most input/output activations in attn
        for key, val in self.quantization_attn_dict.items():
            if key.startswith("quantize_") and val == True:
                self.flash = False
                print("flash attention removed due to Quantization")
                break

        if self.disable_flash_attention:
            self.flash = False

        # Softmax Variant Selection
        self.softmax_variant_attn = config.softmax_variant_attn
        if self.softmax_variant_attn == "softmax":
            # Enable flash attention, which is compatible with 'softmax'
            if self.disable_flash_attention or self.flash == False:
                print("setting non-flash softmax attn")
            else:
                self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
                print("setting flash attn")
        else:
            # Remove flash attention (only compatible with 'softmax')
            print("flash attention removed due to softmax alternative")
            self.flash = False
            # Set softmax_layer_attn to custom softmax alternative
            self.softmax_layer_attn = softmax_dictionary[config.softmax_variant_attn](config)

        if (not self.flash) and (not self.use_flex_attn):
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    # Flex Attention Related
    def sliding_window_causal(self, b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        window_mask = q_idx - kv_idx <= self.window_size
        return causal_mask & window_mask

    def get_block_mask(self, T, device):
        if T not in self.block_masks:
            block_mask = create_block_mask(
                    self.sliding_window_causal,
                    B=None,
                    H=None,
                    Q_LEN=T,
                    KV_LEN=T,
                    device=device
                    )
            self.block_masks[T] = block_mask
        else:
            block_mask = self.block_masks[T]
        return block_mask
    # End Flex Attention Related

    def forward(self, x, iter_num):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        if self.quantization_attn_dict["quantize_attn_act_input"]:
            num_bits = self.quantization_attn_dict["quantize_attn_act_input_bits"]
            quant_method = self.quantization_attn_dict["activations_quant_method"]
            x = fake_quantize_act(self, "attn_act_input", x, num_bits, quant_method, iter_num)

        q = self.c_attn_q(x)
        k = self.c_attn_k(x)
        v = self.c_attn_v(x)

        if self.window_size is not None:
            if self.use_flex_attn is not None:
                self.block_masks = {}
            else:
                self.window_mask = torch.ones((1, 1, T, T), device=x.device)
                self.window_mask = torch.triu(self.window_mask, diagonal=-self.window_size)
                self.window_mask = self.bias[:,:,:T,:T] * self.window_mask

        if self.gate:
            if self.n_kv_group == self.n_head:
                Gating = nn.Linear(self.n_embd, self.n_embd, bias=True, device=x.device)
                gate_ = torch.sigmoid(Gating(x))
                q = q * gate_
                k = k * gate_
                v = v * gate_
            else:
                # TODO: Test more methods to merge Attention Gates with GQA
                # TODO: Evaluate each method's ability to even out parameter sizes
                Gating_q = nn.Linear(self.n_embd, self.n_embd, bias=True, device=x.device)
                Gating_kv = nn.Linear(self.n_embd, self.kv_dim, bias=True, device=x.device)
                gate_qx = Gating_q(x)
                gate_q = torch.sigmoid(gate_qx)
                gate_kv = torch.sigmoid(Gating_kv(gate_qx))
                q = q * gate_q
                k = k * gate_kv
                v = v * gate_kv

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_h, T, hs)
        k = k.view(B, T, self.n_kv_group, C // self.n_head).transpose(1, 2) # (B, n_kv, T, hs)
        v = v.view(B, T, self.n_kv_group, C // self.n_head).transpose(1, 2) # (B, n_kv, T, hs)

        # rotate q and k before evaluating with the heads
        if (self.rotary_emb_q is not None) and (self.rotary_emb_k is not None):
            q = self.rotary_emb_q(q)
            k = self.rotary_emb_k(k)

        y = None
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        elif self.use_flex_attn and self.window_size is not None:
            block_mask = self.get_block_mask(T, x.device)
            y = torch.nn.attention.flex_attention.flex_attention(q, k, v, block_mask=block_mask)
        else:
            if self.quantization_attn_dict["quantize_attn_act_qk_mult_q_input"]:
                num_bits = self.quantization_attn_dict["quantize_attn_act_qk_mult_q_input_bits"]
                quant_method = self.quantization_attn_dict["activations_quant_method"]
                q = fake_quantize_act(self, "attn_act_qk_mult_q_input", q, num_bits, quant_method, iter_num)
            if self.quantization_attn_dict["quantize_attn_act_qk_mult_k_input"]:
                num_bits = self.quantization_attn_dict["quantize_attn_act_qk_mult_k_input_bits"]
                quant_method = self.quantization_attn_dict["activations_quant_method"]
                k = fake_quantize_act(self, "attn_act_qk_mult_k_input", k, num_bits, quant_method, iter_num)

            att = None
            # manual implementation of attention
            if self.n_head != self.n_kv_group:
                k_repeated = k.repeat_interleave(self.n_head // self.n_kv_group, dim=1)
                att = (q @ k_repeated.transpose(-2, -1)) / math.sqrt(k.size(-1))
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            # apply masks
            if self.window_size is not None:
                # add mask for sliding window attention
                att = att.masked_fill(self.window_mask == 0, float('-inf'))
            else:
                # regular lower triangle attention
                att = att.masked_fill(self.bias[:,:,:T,:T].to(x.device) == 0, float('-inf'))

            # fire position embeddings
            if self.use_fire_embeddings is not None:
                # add learned fire bias
                att = att + self.fire_pos_enc(x)

            if self.quantization_attn_dict["quantize_attn_act_softmax_input"]:
                num_bits = self.quantization_attn_dict["quantize_attn_act_softmax_input_bits"]
                quant_method = self.quantization_attn_dict["activations_quant_method"]
                att = fake_quantize_act(self, "attn_act_softmax_input", att, num_bits, quant_method, iter_num, causal_mask=True)

            # softmax variation
            if self.softmax_variant_attn != 'softmax':
                att = self.softmax_layer_attn(att)
            else:
                att = F.softmax(att, dim=-1)

            att = self.attn_dropout(att)

            if self.quantization_attn_dict["quantize_attn_act_pv_mult_p_input"]:
                num_bits = self.quantization_attn_dict["quantize_attn_act_pv_mult_p_input_bits"]
                quant_method = self.quantization_attn_dict["activations_quant_method"]
                att = fake_quantize_act(self, "attn_act_pv_mult_p_input", att, num_bits, quant_method, iter_num)
            if self.quantization_attn_dict["quantize_attn_act_pv_mult_v_input"]:
                num_bits = self.quantization_attn_dict["quantize_attn_act_pv_mult_v_input_bits"]
                quant_method = self.quantization_attn_dict["activations_quant_method"]
                v = fake_quantize_act(self, "attn_act_pv_mult_v_input", v, num_bits, quant_method, iter_num)

            if self.n_head != self.n_kv_group:
                v_repeated = v.repeat_interleave(self.n_head // self.n_kv_group, dim=1)
                y = att @ v_repeated # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            else:
                y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        if self.quantization_attn_dict["quantize_attn_act_pv_mult_output"]:
            num_bits = self.quantization_attn_dict["quantize_attn_act_pv_mult_output_bits"]
            quant_method = self.quantization_attn_dict["activations_quant_method"]
            y = fake_quantize_act(self, "attn_act_pv_mult_output", y, num_bits, quant_method, iter_num)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        if self.quantization_attn_dict["quantize_attn_act_output"]:
            num_bits = self.quantization_attn_dict["quantize_attn_act_output_bits"]
            quant_method = self.quantization_attn_dict["activations_quant_method"]
            y = fake_quantize_act(self, "attn_act_output", y, num_bits, quant_method, iter_num)

        return y

class LinearAttention(nn.Module):
    """ Implements Linear Attention as described in:
    Katharopoulos, A., et al. (2020). Transformers are RNNs:
    Fast Autoregressive Transformers with Linear Attention. ICML.
    https://arxiv.org/abs/2006.16236

    This class replaces the standard softmax attention with a
    kernel-based linear attention mechanism, enabling linear
    time and space complexity with respect to sequence length.
    """
    def __init__(self, config, fire_pos_enc=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head

        # Combined linear layer for q, k, v
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.scale = torch.nn.Parameter(torch.tensor(1.0 / math.sqrt(self.head_size)))


    def forward(self, x, iter_num=None):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_size)
        k = k.view(B, T, self.n_head, self.head_size)
        v = v.view(B, T, self.n_head, self.head_size)

        # NEW: Scale BEFORE the feature map
        q = q * self.scale
        k = k * self.scale

        q = F.elu(q) + 1
        k = F.elu(k) + 1

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        kv = k * v
        k_cumsum = k.cumsum(dim=2)
        kv_cumsum = kv.cumsum(dim=2)


        eps = 1e-3  # Increased epsilon
        y = torch.einsum("BHTD,BHTD->BHTD", q, kv_cumsum) / (torch.einsum("BHTD,BHTD->BHT", q, k_cumsum)[..., None].clamp(min=eps))

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y

class HymbaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        HymbaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class MambaBlock(nn.Module):
    """ This function contains code adapted from [Hymba](https://github.com/NVlabs/hymba/)
    by the NVIDIA team, licensed under the [NVIDIA Open Model License Agreement]
    (https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
    """
    def __init__(self, config, fire_pos_enc=None):
        super().__init__()

        self.d_model = config.n_embd
        self.d_inner = int(self.d_model * config.ssm_mamba_expand)
        self.conv_kernel_size = config.ssm_conv_kernel_size
        self.dt_rank = config.ssm_dt_rank
        self.d_state = config.ssm_d_state
        self.io_bias = config.ssm_io_bias

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=self.conv_kernel_size,
            groups=self.d_inner,
            padding=self.conv_kernel_size - 1
        )

        num_ssm_param = 1
        self.in_proj = nn.ModuleList([nn.Linear(self.d_model, self.d_inner * 2, bias=self.io_bias)])
        self.x_proj = nn.ModuleList([nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False) for _ in range(num_ssm_param)])
        self.dt_proj = nn.ModuleList([nn.Linear(self.dt_rank, self.d_inner, bias=True) for _ in range(num_ssm_param)])
        self.out_proj = nn.ModuleList([nn.Linear(self.d_inner, self.d_model, bias=self.io_bias)])

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.d_inner, -1).contiguous()
        self.A_log = nn.ParameterList([nn.Parameter(torch.log(A)) for _ in range(num_ssm_param)])

        self.D = nn.ParameterList([nn.Parameter(torch.ones(self.d_inner)) for _ in range(num_ssm_param)])

        self.dt_layernorm = HymbaRMSNorm(self.dt_rank, eps=1e-06)
        self.B_layernorm = HymbaRMSNorm(self.d_state, eps=1e-06)
        self.C_layernorm = HymbaRMSNorm(self.d_state, eps=1e-06)
        self.scan_outputs_layernorm = HymbaRMSNorm(self.d_inner)

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x, gate, iter_num=None):
        '''
        Parameters:
            x: (batch_size, seqlen, d_model)
        Return:
            scan_outputs: (batch_size, seqlen, d_model)

        # d_model == n_embd (in attention)
        # d_inner == d_model * mamba_expand

        # conv1d.weight: (d_inner, 1, conv_kernel_size)
        # conv_weights: (d_inner, conv_kernel_size)
        # hidden_states: (batch_size, d_inner, seqlen)
        # gate: (batch_size, d_inner, seqlen)
        # delta: (batch_size, seqlen, dt_rank)
        # discrete_delta: (batch, d_inner, seqlen)
        # A: (d_inner, d_state)
        # B: (batch_size, seqlen, d_state) before transpose(1,2)
        # C: (batch_size, seqlen, d_state) before transpose(1,2)
        '''
        # we only have a single mamba head at this point
        index = 0

        projected_states = self.in_proj[index](x).transpose(1,2)

        hidden_states, gate = projected_states.tensor_split((self.d_inner,), dim=1)

        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        hidden_states = causal_conv1d_fn(
            hidden_states, conv_weights, self.conv1d.bias, activation="silu"
        )

        ssm_parameters = self.x_proj[index](hidden_states.transpose(1, 2))
        delta, B, C = torch.split(ssm_parameters, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta, B, C = self._apply_layernorms(delta, B, C)
        dt_proj_bias = self.dt_proj[index].bias
        self.dt_proj[index].bias = None
        discrete_delta = self.dt_proj[index](delta).transpose(1, 2)  # (batch_size, d_inner, seqlen)
        self.dt_proj[index].bias = dt_proj_bias

        A = -torch.exp(self.A_log[index].float())

        dt_proj_bias = dt_proj_bias.float() if dt_proj_bias is not None else None

        # mammba kernel from mamba_ssm
        outputs = selective_scan_fn(
            hidden_states,                          # (batch_size, d_inner, seqlen)
            discrete_delta,
            A,
            B.transpose(1, 2).to(torch.float16),    # torch.float32 -> torch.float16 for selective_scan_fn
            C.transpose(1, 2).to(torch.float16),    # torch.float32 -> torch.float16 for selective_scan_fn
            self.D[index].float(),
            z=gate,
            delta_bias=dt_proj_bias,
            delta_softplus=True,
            return_last_state=True,
        )                                           # (batch_size, d_inner, seqlen)

        if len(outputs) == 3:
            scan_outputs, _, _ = outputs            # scan_outputs, ssm_state, last_state
                                                    # ssm_state are updated inplace
        else:
            scan_outputs, _ = outputs

        scan_outputs = scan_outputs.transpose(1, 2)  # (batch_size, seqlen, d_inner)
        scan_outputs = self.scan_outputs_layernorm(scan_outputs)

        output = self.out_proj[index](scan_outputs)
        return output

class Identity(nn.Module):
    def __init__(self, config, fire_pos_enc=None):
        super(Identity, self).__init__()

    def forward(self, x, iter_num=None):
        return x

attention_dictionary = {
    "causal": CausalSelfAttention,
    "linear": LinearAttention,
    "ssm": MambaBlock,
    "identity": Identity,
}

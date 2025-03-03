# OLMo_custom_attention.py
import torch
import math
import torch.nn.functional as F
from transformers.models.olmo.configuration_olmo import OlmoConfig
from transformers.models.olmo.modeling_olmo import OlmoAttention, repeat_kv, apply_rotary_pos_emb

class OlmoCustomAttention(OlmoAttention):
    """
    Custom attention as determined by the attention_variant parameter.
    """
    def __init__(self, config, layer_idx, attention_variant, zeta=1.0, gamma=0.0):
        super().__init__(config, layer_idx=layer_idx)
        if attention_variant not in ["clipped", "relu", "relu_norm", "gated", "softplus", "softplus_norm", 
                                     "sigmoid", "sigmoid_norm", "obo", "learned_obo"]:
            raise ValueError("attention_variant isn't one of the valid possible options")
        self.attention_variant = attention_variant
        self.zeta = zeta
        self.gamma = gamma
        # Initialize gating parameters if gated attention is chosen.
        if self.attention_variant == "gated":
            # Each head gets its own linear transformation (d_head -> 1)
            self.gate_weight = torch.nn.Parameter(torch.randn(self.num_heads, self.head_dim) * (1.0 / math.sqrt(self.head_dim)))
            self.gate_bias = torch.nn.Parameter(torch.zeros(self.num_heads, 1))
        elif self.attention_variant == "learned_obo":
            self.obo_param = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_value=None, cache_position=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Project hidden states to query, key, and value.
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.config.clip_qkv is not None:
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        # Apply rotary positional embeddings.
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Expand key and value for key/value groups.
        key_states_exp = repeat_kv(key_states, self.num_key_value_groups)
        value_states_exp = repeat_kv(value_states, self.num_key_value_groups)
        attn_scores = torch.matmul(query_states, key_states_exp.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states_exp.shape[-2]]
            attn_scores = attn_scores + causal_mask

        if self.attention_variant == "clipped":
            # Clipped softmax: first compute softmax...
            attn_weights = F.softmax(attn_scores, dim=-1)
            # ...then apply clipping
            attn_weights = torch.clamp((self.zeta - self.gamma) * attn_weights + self.gamma, min=0.0, max=1.0)
        elif self.attention_variant == "gated":
            # For gated attention, use the standard softmax for computing attn_weights.
            attn_weights = F.softmax(attn_scores, dim=-1)
        elif self.attention_variant == "relu":
            # ReLU-based attention
            attn_weights = F.relu(attn_scores)
        elif self.attention_variant == "relu_norm":
            attn_weights = F.relu(attn_scores)
            # Normalize the weights so they sum to 1 along the key dimension.
            attn_sum = attn_weights.sum(dim=-1, keepdim=True) + 1e-9
            attn_weights = attn_weights / attn_sum
        elif self.attention_variant == "softplus":
            attn_weights = F.softplus(attn_scores)
        elif self.attention_variant == "softplus_norm":
            attn_weights = F.softplus(attn_scores)
            # Normalize so weights sum to 1 along the key dimension.
            attn_sum = attn_weights.sum(dim=-1, keepdim=True) + 1e-9
            attn_weights = attn_weights / attn_sum
        elif self.attention_variant == "sigmoid":
            attn_weights = torch.sigmoid(attn_scores)
        elif self.attention_variant == "sigmoid_norm":
            attn_weights = torch.sigmoid(attn_scores)
            # Normalize so weights sum to 1 along the key dimension.
            attn_sum = attn_weights.sum(dim=-1, keepdim=True) + 1e-9
            attn_weights = attn_weights / attn_sum
        elif self.attention_variant == "obo":
            exp_scores = torch.exp(attn_scores)
            denom = 1.0 + exp_scores.sum(dim=-1, keepdim=True)
            attn_weights = exp_scores / denom
        elif self.attention_variant == "learned_obo":
            exp_scores = torch.exp(attn_scores)
            denom = self.obo_param + exp_scores.sum(dim=-1, keepdim=True)
            attn_weights = exp_scores / denom
        else:
            raise ValueError("attention_variant isn't one of the valid possible options")

        attn_weights = F.dropout(attn_weights, p=self.attention_dropout if self.training else 0.0, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states_exp)

        # If using gated attention, compute gating values and apply them.
        if self.attention_variant == "gated":
            # Compute gate scores per head and token:
            # query_states has shape (B, nheads, T, d_head)
            gate_scores = torch.einsum('bhtd,hd->bht', query_states, self.gate_weight) + self.gate_bias.squeeze(-1)
            gate_values = torch.sigmoid(gate_scores)  # shape: (B, nheads, T)
            gate_values = gate_values.unsqueeze(-1)   # shape: (B, nheads, T, 1)
            attn_output = attn_output * gate_values

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

def apply_custom_attention(model, attention_variant, zeta=1.0, gamma=0.0):
    """
    Replaces all decoder layer attention modules in the model with OlmoCustomAttention.
    """
    for layer_idx, decoder_layer in enumerate(model.model.layers):
        original_attn = decoder_layer.self_attn
        custom_attn = OlmoCustomAttention(
            model.config,
            layer_idx=layer_idx,
            attention_variant=attention_variant,
            zeta=zeta,
            gamma=gamma
        )
        custom_attn.load_state_dict(original_attn.state_dict())
        decoder_layer.self_attn = custom_attn
    return model

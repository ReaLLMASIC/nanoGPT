import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import necessary functions from the LLaMA implementation.
# Adjust the import paths if needed.
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb
from transformers.models.llama.modeling_llama import eager_attention_forward  # reference version

# Define our custom eager attention function that uses ReLU instead of softmax.
def custom_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    dropout: float,
    scaling: float,
    **kwargs,
):
    # Compute repeated keys/values if needed (the original version may use a helper; here we assume no change)
    # Compute raw attention scores.
    attn_scores = torch.matmul(query, key.transpose(-1, -2)) * scaling

    if attention_mask is not None:
        # The original code slices the attention_mask appropriately.
        attn_scores = attn_scores + attention_mask

    # Replace softmax with ReLU.
    attn_weights = torch.relu(attn_scores)
    # Manually normalize: sum over last dimension and divide.
    attn_sum = attn_weights.sum(dim=-1, keepdim=True)
    # Avoid division by zero.
    attn_sum[attn_sum == 0] = 1.0
    attn_weights = attn_weights / attn_sum

    # Apply dropout.
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    # Compute the attention output.
    attn_output = torch.matmul(attn_weights, value)
    # In the original, attention output is transposed and contiguously reshaped by the caller.
    return attn_output, attn_weights

# Subclass LlamaAttention to override its forward behavior.
class CustomLlamaAttention(LlamaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: torch.Tensor,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        # Save the input shape.
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Project hidden states to query, key, and value.
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Unpack rotary embeddings and apply them.
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # Update key and value states if caching is used.
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Instead of using the default attention_interface, we use our custom one.
        attn_output, attn_weights = custom_eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # Reshape output back to the original hidden state shape.
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

def replace_llama_attention(model: AutoModelForCausalLM):
    """
    Traverse the model's decoder layers and replace each LlamaAttention module in self_attn
    with our CustomLlamaAttention.
    """
    for i, layer in enumerate(model.model.layers):
        orig_attn = layer.self_attn
        # Instantiate our custom attention module with the same configuration.
        custom_attn = CustomLlamaAttention(orig_attn.config)
        # Copy weights from the original attention module.
        custom_attn.load_state_dict(orig_attn.state_dict())
        # Replace the attention module.
        layer.self_attn = custom_attn
        print(f"Replaced self_attn in layer {i} with CustomLlamaAttention.")

# --- Usage Example ---
model_name = "llama-7B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Replace attention modules with the custom version.
replace_llama_attention(model)

# Run a test forward pass to verify.
input_text = "This is a test input to verify custom ReLU attention."
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
model.eval()
with torch.no_grad():
    outputs = model(**inputs)

print("Forward pass completed. Check the console for any debug prints if added in custom attention.")

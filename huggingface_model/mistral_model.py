import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the original eager attention function from Mistral's modeling file.
# Adjust the import path if needed.
from transformers.models.mistral import modeling_mistral as mistral_mod

# 1. Define a custom attention function that uses ReLU instead of softmax.
def custom_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # Repeat keys and values if needed (same as original)
    key_states = mistral_mod.repeat_kv(key, module.num_key_value_groups)
    value_states = mistral_mod.repeat_kv(value, module.num_key_value_groups)

    # Compute raw attention scores and apply scaling.
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        # Here we assume attention_mask is already of proper shape (or we use slicing as in original)
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # **Replace softmax with ReLU**
    attn_weights = torch.relu(attn_weights)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    
    # Compute attention output.
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights

# 2. Monkey-patch the eager_attention_forward function in Mistral's module.
mistral_mod.eager_attention_forward = custom_eager_attention_forward
print("Replaced Mistral's eager_attention_forward with custom version using ReLU.")

# 3. Force the configuration to use "eager" attention implementation.
# This ensures that the forward method in MistralAttention calls our custom function.
model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, torch_dtype=torch.float16
)
model.config._attn_implementation = "eager"  # Ensure the eager path is used.
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 4. Run a small forward pass to test.
input_text = "This is a test input for the custom attention."
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
model.eval()
with torch.no_grad():
    outputs = model(**inputs)

print(model)

print("Forward pass completed. Custom ReLU attention should have been used.")
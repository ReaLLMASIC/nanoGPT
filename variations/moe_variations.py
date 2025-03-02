# variations/moe_variations.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from variations.router_variations import router_dictionary
from variations.mlp_variations import get_mlp_instance


class MoELayer(nn.Module):
    """Mixture of Experts layer to replace FFN (or every other FFN)."""

    def __init__(self, config):
        super().__init__()
        self.top_k = config.moe_top_k
        # TODO: implement expert capacity throttling
        # self.expert_capacity = config.expert_capacity
        self.num_experts = config.n_experts
        self.router = router_dictionary[config.moe_router_scheme](config)
        self.experts = nn.ModuleList([get_mlp_instance(config) for _ in range(config.n_experts)])

    def forward(self, x, iter_num=None):
        # Assuming x has shape [batch_size, seq_len, n_embd]
        batch_size, seq_len, _ = x.shape
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        # Flatten the batch and sequence dimensions to treat each token independently
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output


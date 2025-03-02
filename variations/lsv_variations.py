# variation/lsv_variations.py
import torch
import torch.nn as nn

# Helper Mixin factoring out the freezing process
class FreezeNonSelectedMixin:
    def freeze_non_selected_rows(self, lsv_matrix, lsv_index):
        """
        Freezes all rows in lsv_matrix that are not selected by lsv_index.
        """
        with torch.no_grad():
            for i in range(lsv_matrix.size(0)):
                if i == lsv_index:
                    lsv_matrix[i].requires_grad = True  # Enable gradient update for selected row
                else:
                    lsv_matrix[i].requires_grad = False  # Freeze other rows

class LSVBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lsv_index = 0    # Dataset index default to zero
        self.lsv_dataset_num = config.lsv_dataset_num
        self.lsv_embd_dim = config.n_embd
        self.lsv_scaling_factor = 1.0
        self.mode = 1
        self.mixture = []

    def update_lsv_scaling_factor(self, new_scaling_factor):
        self.lsv_scaling_factor = new_scaling_factor
        # print("set scaling factor to:", self.lsv_scaling_factor)

    def get_lsv_scaling_factor(self):
        return self.lsv_scaling_factor

    def update_lsv_index(self, new_index):
        self.lsv_index = new_index
        # print("new index", self.lsv_index)

    def set_mixture(self, mixture_list):
        """ for variation to override """
        self.mixture = mixture_list
        pass

    def set_mode(self, mode):
        """ Modes, generally:
        1 = one hot
        2 = mixture mode (set mixture and forget)
        """
        self.mode = mode

    def forward(self, x):
        return x

class OneHotLSV(LSVBase, FreezeNonSelectedMixin):
    def __init__(self, config):
        # Initialize the base class
        super().__init__(config)

        # Initialize the lsv_matrix and one-hot vector
        self.lsv_matrix = nn.Parameter(torch.empty(self.lsv_dataset_num, config.n_embd, device=self.device))
        torch.nn.init.normal_(self.lsv_matrix, mean=0.00, std=0.02)
        self.one_hot_vector = torch.zeros(self.lsv_matrix.size(0), device=self.device)
        self.mode = 1

    def set_mixture(self, mixture_list):
        """ mixture of different vectors """
        for i in range(len(mixture_list)):
            self.one_hot_vector[i] = mixture_list[i]
        print("mixture set to:", self.one_hot_vector)

    def forward(self, x):
        # Freeze all rows that are not selected by the one-hot vector
        self.freeze_non_selected_rows(self.lsv_matrix, self.lsv_index)

        # Create a one-hot vector for the dataset index
        if self.mode == 1:
            self.one_hot_vector.zero_()  # Reset the one-hot vector
            self.one_hot_vector[self.lsv_index] = 1.0 * self.lsv_scaling_factor

        # Multiply the one-hot vector by the learned parameter matrix to get the selected vector
        selected_vector = torch.matmul(self.one_hot_vector, self.lsv_matrix)

        # Add the selected vector to the input tensor x
        x = x + selected_vector

        return x

class LinearCombinationLSV(LSVBase, FreezeNonSelectedMixin):
    def __init__(self, config):
        # Initialize the base class
        super().__init__(config)

        # Initialize the lsv_matrix and learned combination weights
        self.lsv_matrix = nn.Parameter(torch.empty(self.lsv_dataset_num, config.n_embd, device=self.device))
        torch.nn.init.normal_(self.lsv_matrix, mean=0.00, std=0.02)

        # Learnable linear combination vector
        self.linear_comb_matrix = nn.Parameter(torch.empty(self.lsv_dataset_num, self.lsv_dataset_num, device=self.device))
        torch.nn.init.normal_(self.linear_comb_matrix, mean=0.00, std=0.02)

    def forward(self, x):

        # Only learn target dataset linear comb, and dataset vector
        self.freeze_non_selected_rows(self.lsv_matrix, self.lsv_index)
        self.freeze_non_selected_rows(self.linear_comb_matrix, self.lsv_index)

        self.one_hot_vector.zero_()  # Reset the one-hot vector
        self.one_hot_vector[self.lsv_index] = 1.0 * self.lsv_scaling_factor

        selected_linear_comb_vector = torch.matmul(self.one_hot_vector, self.linear_comb_matrix)

        # Use the learned combination vector instead of a one-hot vector
        combined_vector = torch.matmul(selected_linear_comb_vector, self.lsv_matrix)

        # Add the combined vector to the input tensor x
        x = x + combined_vector

        return x

class OneHotMLPLSV_Gating(LSVBase):
    """OneHotMLPLSV with gating mechanism using sigmoid activation."""
    def __init__(self, config):
        super().__init__(config)
        mlp_width = 64
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.n_embd, mlp_width),
                nn.ReLU(),
                nn.Linear(mlp_width, mlp_width),
                nn.ReLU(),
                nn.Linear(mlp_width, config.n_embd)
            ).to(self.device) for _ in range(self.lsv_dataset_num)
        ])

        # Router produces gating values between 0 and 1 for each MLP
        self.router = nn.Sequential(
            nn.Linear(config.n_embd, self.lsv_dataset_num),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, seq_length, n_embd = x.size()
        x_flat = x.view(-1, n_embd)

        # Gating values for each MLP
        gates = self.router(x_flat)  # [batch_size * seq_length, lsv_dataset_num]

        combined_output = torch.zeros_like(x_flat)
        for i, mlp in enumerate(self.mlps):
            mlp_output = mlp(x_flat)
            gate = gates[:, i].unsqueeze(-1)
            combined_output += gate * mlp_output

        combined_output = combined_output.view(batch_size, seq_length, n_embd)
        x = x + combined_output
        return x


class MixtureOfLSV(LSVBase):
    """ A FIRE Inspired method for combining LSVs with a learned router. """
    def __init__(self, config):
        # Initialize the base class
        super().__init__(config)

        # MLP configuration
        mlp_width = 64
        self.mlps = nn.ModuleList()

        # Create an MLP for each index
        for _ in range(self.lsv_dataset_num):
            mlp_layers = []
            mlp_layers.append(nn.Linear(config.n_embd, mlp_width))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Linear(mlp_width, mlp_width))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Linear(mlp_width, config.n_embd))  # Output is embedding dimension size
            self.mlps.append(nn.Sequential(*mlp_layers).to(self.device))

        # Define the learned router, which will output a probability distribution over MLPs
        self.router = nn.Sequential(
            nn.Linear(config.n_embd, self.lsv_dataset_num),
            nn.Softmax(dim=-1)  # Output a probability distribution over MLPs
        )

    def forward(self, x):
        # x: [batch_size, seq_length, n_embd]

        # Flatten x to merge batch and sequence dimensions for the router and MLPs
        batch_size, seq_length, n_embd = x.size()
        x_flat = x.view(-1, n_embd)  # [batch_size * seq_length, n_embd]

        # Get the router's output: a probability distribution over the MLPs
        router_probs = self.router(x_flat)  # [batch_size * seq_length, lsv_dataset_num]

        # Compute the output as a weighted sum of MLP outputs
        combined_output = torch.zeros_like(x_flat, device=self.device)  # [batch_size * seq_length, n_embd]

        for i, mlp in enumerate(self.mlps):
            mlp_output = mlp(x_flat)  # [batch_size * seq_length, n_embd]

            # Get the router probability for the current MLP
            prob = router_probs[:, i].unsqueeze(-1)  # [batch_size * seq_length, 1]

            # Multiply and accumulate
            combined_output += prob * mlp_output  # [batch_size * seq_length, n_embd]

        # Reshape combined_output back to [batch_size, seq_length, n_embd]
        combined_output = combined_output.view(batch_size, seq_length, n_embd)

        # Combine the MLP output with x (here we just add them)
        x = x + combined_output

        return x


class OneHotMLPLSV(LSVBase):
    """ A FIRE Inspired method for combining LSVs """
    def __init__(self, config):
        # Initialize the base class
        super().__init__(config)

        # Create multiple MLPs, one for each index
        mlp_width = 64
        self.mlps = nn.ModuleList()

        # Create a tensor containing the constant input "1"
        self.constant_input = torch.tensor([[1.0]], device=self.device)

        for _ in range(self.lsv_dataset_num):
            mlp_layers = []
            mlp_layers.append(nn.Linear(config.n_embd, mlp_width))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Linear(mlp_width, mlp_width))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Linear(mlp_width, config.n_embd))  # Output is embedding dimension size
            self.mlps.append(nn.Sequential(*mlp_layers).to(self.device))

    def freeze_non_selected_mlps(self):
        """
        Freezes all MLPs except the one corresponding to the current lsv_index.
        """
        # Freeze all MLPs
        for i, mlp in enumerate(self.mlps):
            for param in mlp.parameters():
                param.requires_grad = False

        # Unfreeze the selected MLP
        for param in self.mlps[self.lsv_index].parameters():
            param.requires_grad = True

    def forward(self, x):

        # Select the MLP based on the index
        if self.mode == 1:
            # Freeze all non-selected MLPs and unfreeze the selected one
            self.freeze_non_selected_mlps()
            selected_mlp = self.mlps[self.lsv_index]
            # Pass the constant input through the selected MLP
            mlp_output = selected_mlp(x)
        else:
            mlp_output = 0
            for i in range(len(self.mlps)):
                mlp_output += self.mlps[i](x) * self.mixture[i]

        # Combine the MLP output with x (you can combine it in different ways, here we just add them)
        x = x + mlp_output

        return x

class OneHotMLPLSV_TopK(LSVBase):
    """OneHotMLPLSV with Top-K selection of MLPs."""
    def __init__(self, config, k=2):
        super().__init__(config)
        self.k = k
        mlp_width = 64
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.n_embd, mlp_width),
                nn.ReLU(),
                nn.Linear(mlp_width, config.n_embd)
            ).to(self.device) for _ in range(self.lsv_dataset_num)
        ])

        self.router = nn.Linear(config.n_embd, self.lsv_dataset_num)

    def forward(self, x):
        batch_size, seq_length, n_embd = x.size()
        x_flat = x.view(-1, n_embd)

        router_logits = self.router(x_flat)
        topk_values, topk_indices = torch.topk(router_logits, self.k, dim=-1)
        mask = torch.zeros_like(router_logits)
        mask.scatter_(1, topk_indices, 1.0)

        # Use straight-through estimator
        gates = (mask - router_logits.detach()) + router_logits

        combined_output = torch.zeros_like(x_flat)
        for i, mlp in enumerate(self.mlps):
            gate = gates[:, i].unsqueeze(-1)
            if gate.abs().sum() > 0:
                mlp_output = mlp(x_flat)
                combined_output += gate * mlp_output

        combined_output = combined_output.view(batch_size, seq_length, n_embd)
        x = x + combined_output
        return x

class OneHotMLPLSV_MoE(LSVBase):
    """OneHotMLPLSV using Mixture of Experts with learned routing."""
    def __init__(self, config, router_temperature=1.0):
        super().__init__(config)
        self.router_temperature = router_temperature
        mlp_width = 64
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.n_embd, mlp_width),
                nn.ReLU(),
                nn.Linear(mlp_width, config.n_embd)
            ).to(self.device) for _ in range(self.lsv_dataset_num)
        ])

        self.router = nn.Linear(config.n_embd, self.lsv_dataset_num)

    def forward(self, x):
        batch_size, seq_length, n_embd = x.size()
        x_flat = x.view(-1, n_embd)

        router_logits = self.router(x_flat) / self.router_temperature
        router_probs = nn.functional.softmax(router_logits, dim=-1)

        combined_output = torch.zeros_like(x_flat)
        for i, mlp in enumerate(self.mlps):
            mlp_output = mlp(x_flat)
            prob = router_probs[:, i].unsqueeze(-1)
            combined_output += prob * mlp_output

        combined_output = combined_output.view(batch_size, seq_length, n_embd)
        x = x + combined_output
        return x

class OneHotMLPLSV_Attention(LSVBase):
    """OneHotMLPLSV with attention-based routing."""
    def __init__(self, config):
        super().__init__(config)
        mlp_width = 64
        self.lsv_dataset_num = config.lsv_dataset_num
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.n_embd, mlp_width),
                nn.ReLU(),
                nn.Linear(mlp_width, config.n_embd)
            ).to(self.device) for _ in range(self.lsv_dataset_num)
        ])

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(self.lsv_dataset_num, config.n_embd))

    def forward(self, x):
        batch_size, seq_length, n_embd = x.size()
        x_flat = x.view(-1, n_embd)

        # Compute attention scores
        attention_scores = torch.matmul(x_flat, self.queries.t())
        router_probs = nn.functional.softmax(attention_scores, dim=-1)

        combined_output = torch.zeros_like(x_flat)
        for i, mlp in enumerate(self.mlps):
            mlp_output = mlp(x_flat)
            prob = router_probs[:, i].unsqueeze(-1)
            combined_output += prob * mlp_output

        combined_output = combined_output.view(batch_size, seq_length, n_embd)
        x = x + combined_output
        return x



lsv_dictionary = {
    "one_hot": OneHotLSV,
    "linear_comb": LinearCombinationLSV,
    "one_hot_mlp": OneHotMLPLSV,
    "ohmg": OneHotMLPLSV_Gating,
    "ohmt": OneHotMLPLSV_TopK,
    "ohmm": OneHotMLPLSV_MoE,
    "ohma": OneHotMLPLSV_Attention,
    "mol": MixtureOfLSV,
}

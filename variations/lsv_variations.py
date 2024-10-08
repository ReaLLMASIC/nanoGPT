
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

    def update_lsv_scaling_factor(self, new_scaling_factor):
        self.lsv_scaling_factor = new_scaling_factor

    def update_lsv_index(self, new_index):
        self.lsv_index = new_index

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

    def forward(self, x):
        # Freeze all rows that are not selected by the one-hot vector
        self.freeze_non_selected_rows(self.lsv_matrix, self.lsv_index)

        # Create a one-hot vector for the dataset index
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

        # One hot for selecting linear_combination
        self.one_hot_vector = torch.zeros(self.lsv_dataset_num, device=self.device)

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

class OneHotMLPLSV(LSVBase):
    """ A FIRE Inspired method for combining LSVs """
    def __init__(self, config):
        # Initialize the base class
        super().__init__(config)

        # Create multiple MLPs, one for each index
        mlp_width = 32
        self.mlps = nn.ModuleList()

        # Create a tensor containing the constant input "1"
        self.constant_input = torch.tensor([[1.0]], device=self.device)

        for _ in range(self.lsv_dataset_num):
            mlp_layers = []
            mlp_layers.append(nn.Linear(1, mlp_width))
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
        # Freeze all non-selected MLPs and unfreeze the selected one
        self.freeze_non_selected_mlps()

        # Select the MLP based on the index
        selected_mlp = self.mlps[self.lsv_index]

        # Pass the constant input through the selected MLP
        mlp_output = selected_mlp(self.constant_input)

        # Compute the result of the selected MLP
        mlp_output = selected_mlp(x)

        # Combine the MLP output with x (you can combine it in different ways, here we just add them)
        x = x + mlp_output

        return x

class RunningAverageLinearCombinationLSV(LSVBase, FreezeNonSelectedMixin):
    def __init__(self, config, ema_alpha=0.00001526):
        # Initialize the base class
        super().__init__(config)

        # Initialize running averages for each dataset index using EMA
        self.running_averages = torch.zeros(self.lsv_dataset_num, config.n_embd, device=self.device)
        self.ema_alpha = ema_alpha  # Smoothing factor for EMA

        # Learnable linear combination matrix (like before)
        self.linear_comb_matrix = nn.Parameter(torch.empty(self.lsv_dataset_num, self.lsv_dataset_num, device=self.device))
        torch.nn.init.normal_(self.linear_comb_matrix, mean=0.00, std=0.02)

    def update_running_average(self, x):
        """
        Update the EMA running average for the current lsv_index using the input x.
        x has shape (batch_size, context_length, n_embd), and we compute the mean across batch and context length.
        """
        # Compute the mean across all dimensions except the last (-1), i.e., batch and context length dimensions
        batch_context_mean = x.mean(dim=(-2, -3))  # Averaging across batch and context dimensions

        # Update the running average for the selected index using EMA
        current_average = self.running_averages[self.lsv_index]
        new_average = self.ema_alpha * batch_context_mean + (1 - self.ema_alpha) * current_average

        # Store the updated average
        self.running_averages[self.lsv_index] = new_average

    def forward(self, x):
        # Update the running average for the selected lsv_index
        self.update_running_average(x)

        # freeze unused rows of the linear_comb matrix
        self.freeze_non_selected_rows(self.linear_comb_matrix, self.lsv_index)

        # Use the linear combination method from the previous strategy
        one_hot_vector = torch.zeros(self.lsv_dataset_num, device=x.device)
        one_hot_vector[self.lsv_index] = 1.0 * self.lsv_scaling_factor

        # Select the combination weights
        selected_linear_comb_vector = torch.matmul(one_hot_vector, self.linear_comb_matrix)

        # Perform the linear combination using the running averages
        combined_vector = torch.matmul(selected_linear_comb_vector, self.running_averages)

        # Combine the running average linear combination result with x
        x = x + combined_vector

        return x


lsv_dictionary = {
    "one_hot": OneHotLSV,
    "linear_comb": LinearCombinationLSV,
    "one_hot_mlp": OneHotMLPLSV,
    "avg_linear_comb": RunningAverageLinearCombinationLSV,
}

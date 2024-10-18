import torch
import torch.nn as nn

# Custom Activation Variations
class SquaredReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.pow(torch.relu(x), 2)

class PiecewiseLearnableActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_of_points = 7

        # Initialize learnable parameters for x and y values of intermediate points
        self.x_vals = nn.Parameter(torch.linspace(-2, 2, self.num_of_points + 2)[1:-1])  # Exclude -2 and 2

        # Initialize y_vals using GELU output for corresponding x_vals
        gelu = nn.GELU()
        self.y_vals = nn.Parameter(gelu(self.x_vals))

    def forward(self, x):
        # Create a piecewise linear function
        result = torch.zeros_like(x)

        # Leftmost segment (-2 <= x < x_vals[0])
        result = torch.where(x < self.x_vals[0], 0, result)  # x = -2 -> y = 0

        # Intermediate segments (x_vals[i] <= x < x_vals[i+1])
        for i in range(self.num_of_points - 1):
            slope = (self.y_vals[i + 1] - self.y_vals[i]) / (self.x_vals[i + 1] - self.x_vals[i])
            intercept = self.y_vals[i] - slope * self.x_vals[i]
            segment = slope * x + intercept
            result = torch.where((x >= self.x_vals[i]) & (x < self.x_vals[i + 1]), segment, result)

        # Segment before the last (x_vals[-1] <= x < 2)
        slope = (2 - self.y_vals[-1]) / (2 - self.x_vals[-1])
        intercept = self.y_vals[-1] - slope * self.x_vals[-1]
        segment = slope * x + intercept
        result = torch.where((x >= self.x_vals[-1]) & (x < 2), segment, result)

        # Rightmost segment (x >= 2)
        result = torch.where(x >= 2, x, result)  # x = y for x >= 2

        return result

class PiecewiseFullyLearnableActivation(nn.Module):
    def __init__(self, left_bound=-100, right_bound=100, num_of_points=200):
        super().__init__()
        self.num_of_points = num_of_points
        self.left_bound = left_bound
        self.right_bound = right_bound

        # Initialize learnable parameters for x and y values of intermediate points
        self.x_vals = nn.Parameter(torch.linspace(self.left_bound, self.right_bound, self.num_of_points + 2)[1:-1])  # Exclude left_bound and right_bound

        # Initialize y_vals using GELU output for corresponding x_vals
        gelu = nn.GELU()
        self.y_vals = nn.Parameter(gelu(self.x_vals))

    def forward(self, x):
        # Create a piecewise linear function
        result = torch.zeros_like(x)

        # Leftmost segment (left_bound <= x < x_vals[0])
        result = torch.where(x < self.x_vals[0], 0, result)  # x = left_bound -> y = 0

        # Intermediate segments (x_vals[i] <= x < x_vals[i+1])
        for i in range(self.num_of_points - 1):
            slope = (self.y_vals[i + 1] - self.y_vals[i]) / (self.x_vals[i + 1] - self.x_vals[i])
            intercept = self.y_vals[i] - slope * self.x_vals[i]
            segment = slope * x + intercept
            result = torch.where((x >= self.x_vals[i]) & (x < self.x_vals[i + 1]), segment, result)

        # Segment before the last (x_vals[-1] <= x < right_bound)
        slope = (self.right_bound - self.y_vals[-1]) / (self.right_bound - self.x_vals[-1])
        intercept = self.y_vals[-1] - slope * self.x_vals[-1]
        segment = slope * x + intercept
        result = torch.where((x >= self.x_vals[-1]) & (x < self.right_bound), segment, result)

        # Rightmost segment (x >= right_bound)
        result = torch.where(x >= self.right_bound, x, result)  # x = y for x >= right_bound

        return result


class LearnedSplineActivation(nn.Module):
    def __init__(self, num_knots=10, init_x_range=(-2, 2)):
        super().__init__()
        self.num_knots = num_knots

        # Initialize learnable x_vals and y_vals
        x_init = torch.linspace(init_x_range[0], init_x_range[1], num_knots)
        gelu = nn.GELU()
        y_init = gelu(x_init)

        self.x_vals = nn.Parameter(x_init)
        self.y_vals = nn.Parameter(y_init)

    def forward(self, x):
        # Compute spline coefficients
        coeffs = self._compute_spline_coefficients(x_vals, y_vals)

        # Evaluate spline at input x
        result = self._evaluate_spline(x, x_vals_unique, coeffs)
        return result

    def _compute_spline_coefficients(self, x_vals, y_vals):
        """
        Compute the coefficients for cubic spline interpolation.
        """
        n = x_vals.size(0)
        h = x_vals[1:] - x_vals[:-1]  # Intervals between x knots

        # Set up the system of equations
        # Compute the differences in y_vals
        delta = (y_vals[1:] - y_vals[:-1]) / h

        # Construct the tridiagonal matrix
        A = torch.zeros(n, n, device=x_vals.device)
        rhs = torch.zeros(n, device=x_vals.device)

        A[0, 0] = 1  # Natural spline boundary condition
        A[-1, -1] = 1  # Natural spline boundary condition

        for i in range(1, n - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            rhs[i] = 3 * (delta[i] - delta[i - 1])

        # Solve for the second derivatives (M)
        M = torch.linalg.solve(A, rhs)

        # Compute spline coefficients for each interval
        coeffs = []
        for i in range(n - 1):
            h_i = h[i]
            a = y_vals[i]
            b = delta[i] - h_i * (2 * M[i] + M[i + 1]) / 3
            c = M[i] / 2
            d = (M[i + 1] - M[i]) / (6 * h_i)
            coeffs.append((a, b, c, d))

        # Stack coefficients for vectorized computation
        coeffs = tuple(torch.stack(c) for c in zip(*coeffs))
        return coeffs

    def _evaluate_spline(self, x, x_vals, coeffs):
        """
        Evaluate the spline at the input x using the computed coefficients.
        """
        a, b, c, d = coeffs  # Unpack coefficients
        n = x_vals.size(0)

        # Find the interval each x belongs to
        indices = torch.searchsorted(x_vals, x) - 1
        indices = torch.clamp(indices, 0, n - 2)

        x_k = x_vals[indices]
        dx = x - x_k

        # Compute the spline value
        result = a[indices] + b[indices] * dx + c[indices] * dx**2 + d[indices] * dx**3
        return result


activation_dictionary = {
    "celu": nn.CELU(),
    "elu": nn.ELU(),
    "gelu": nn.GELU(),
    "glu": nn.GLU(),
    "leaky_relu": nn.LeakyReLU(),
    "mish": nn.Mish(),
    "piecewise": PiecewiseLearnableActivation(),
    "pfla": PiecewiseFullyLearnableActivation(),
    "learned_spline": LearnedSplineActivation(),
    "prelu": nn.PReLU(),
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
    "rrelu": nn.RReLU(),
    "selu": nn.SELU(),
    "sigmoid": nn.Sigmoid(),
    "silu": nn.SiLU(),
    "softplus": nn.Softplus(),
    "softsign": nn.Softsign(),
    "squared_relu": SquaredReLU(),
    "tanh": nn.Tanh(),
}

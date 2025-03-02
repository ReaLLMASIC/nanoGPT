import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# -----------------------------
# Utility: JL Approximate Multiplication
# -----------------------------
def jl_approx_matmul(A, B, k):
    """
    Computes the JL approximation for the product A * B.
    
    A: tensor with shape (..., d)
    B: tensor with shape (d, ...)
    k: target projection dimension
    """
    d = A.shape[-1]
    # Create a random Gaussian matrix R with shape (k, d), scaled by 1/sqrt(k)
    R = torch.randn(k, d) / np.sqrt(k)
    # Project: A @ R.t() reduces last dim from d to k, R @ B reduces first dim from d to k.
    A_proj = A @ R.t()
    B_proj = R @ B
    return A_proj @ B_proj

##########################################
# Part 1: Matrix * Matrix Analysis
##########################################
print("=== Matrix * Matrix Analysis ===")
# Define square matrix sizes for demonstration.
matrix_sizes = [100, 200, 500, 1000]
error_dict = {}
time_dict = {}

for d in matrix_sizes:
    m = d
    n = d
    A = torch.randn(m, d)
    B = torch.randn(d, n)
    C_exact = A @ B
    # Define a range of k values from 10 up to d (20 points).
    k_values = np.linspace(10, d, 20, dtype=int)
    errors_k = []
    times_k = []
    for k in k_values:
        start = time.time()
        C_approx = jl_approx_matmul(A, B, k)
        t = time.time() - start
        times_k.append(t)
        # Compute the relative Frobenius norm error.
        err = torch.norm(C_exact - C_approx, p='fro') / torch.norm(C_exact, p='fro')
        errors_k.append(err.item())
    error_dict[d] = (k_values, errors_k)
    time_dict[d] = (k_values, times_k)
    print(f"Matrix {d}x{d}: k from {k_values[0]} to {k_values[-1]}, final relative error: {errors_k[-1]:.4f}")

# Combined Plot 1: Relative error vs. projection dimension k (matrix sizes).
plt.figure(figsize=(8, 5))
for d in matrix_sizes:
    k_vals, err_vals = error_dict[d]
    plt.plot(k_vals, err_vals, marker='o', linestyle='-', label=f"{d}x{d}")
plt.xlabel('Projection Dimension k')
plt.ylabel('Relative Frobenius Norm Error')
plt.title('Matrix Multiplication: Error vs. Projection Dimension k')
plt.legend(title="Matrix Size")
plt.grid(True)
plt.savefig('jl_error_vs_k_all.png')
plt.show()

# Combined Plot 2: Computation time vs. projection dimension k (matrix sizes).
plt.figure(figsize=(8, 5))
for d in matrix_sizes:
    k_vals, time_vals = time_dict[d]
    plt.plot(k_vals, time_vals, marker='o', linestyle='-', label=f"{d}x{d}")
plt.xlabel('Projection Dimension k')
plt.ylabel('Computation Time (seconds)')
plt.title('Matrix Multiplication: Time vs. Projection Dimension k')
plt.legend(title="Matrix Size")
plt.grid(True)
plt.savefig('jl_time_vs_k_all.png')
plt.show()

##########################################
# Part 2: Vector * Matrix Analysis (Fixed Vector Size)
##########################################
print("\n=== Vector * Matrix Analysis (Fixed Vector Size) ===")
d_fixed = 500  # original dimension
n_fixed = 500  # number of columns in matrix M
# Create a random vector and matrix.
v = torch.randn(d_fixed)             # shape: (d,)
M = torch.randn(d_fixed, n_fixed)      # shape: (d, n)
v_row = v.unsqueeze(0)                 # shape: (1, d)
y_exact = v_row @ M                    # exact product, shape: (1, n)

# Scan over different projection dimensions k for the fixed vector.
k_values_vec = np.linspace(10, d_fixed, 20, dtype=int)
errors_vec = []
times_vec = []

for k in k_values_vec:
    start = time.time()
    # Compute the JL approximation for y = v^T * M.
    R = torch.randn(k, d_fixed) / np.sqrt(k)
    y_approx = (v_row @ R.t()) @ (R @ M)
    t_k = time.time() - start
    times_vec.append(t_k)
    err_vec = torch.norm(y_exact - y_approx, p='fro') / torch.norm(y_exact, p='fro')
    errors_vec.append(err_vec.item())
    print(f"  k: {k:3d}, Relative error: {err_vec.item():.4f}, Time: {t_k:.6f}s")

# Plot: Relative error vs. k for vector * matrix (fixed vector size).
plt.figure(figsize=(8, 5))
plt.plot(k_values_vec, errors_vec, marker='o', linestyle='-', color='g')
plt.xlabel('Projection Dimension k')
plt.ylabel('Relative Frobenius Norm Error')
plt.title('Vector * Matrix (Fixed Size): Error vs. Projection Dimension k')
plt.grid(True)
plt.savefig('jl_vecmat_error_vs_k.png')
plt.show()

# Plot: Computation time vs. k for vector * matrix (fixed vector size).
plt.figure(figsize=(8, 5))
plt.plot(k_values_vec, times_vec, marker='o', linestyle='-', color='m')
plt.xlabel('Projection Dimension k')
plt.ylabel('Computation Time (seconds)')
plt.title('Vector * Matrix (Fixed Size): Time vs. Projection Dimension k')
plt.grid(True)
plt.savefig('jl_vecmat_time_vs_k.png')
plt.show()

##########################################
# Part 3: Vector * Matrix Analysis (Vector Size Sweep)
##########################################
print("\n=== Vector * Matrix Analysis (Vector Size Sweep) ===")
# Define a list of vector sizes.
vector_sizes = [100, 200, 500, 1000]
error_dict_vec = {}
time_dict_vec = {}

for d_vec in vector_sizes:
    n_vec = d_vec  # Use a square matrix M for simplicity.
    v = torch.randn(d_vec)            # vector of size d_vec.
    M = torch.randn(d_vec, n_vec)       # matrix of shape (d_vec x n_vec).
    v_row = v.unsqueeze(0)              # shape: (1, d_vec)
    y_exact = v_row @ M                 # exact product.
    
    # Define a range of k values from 10 up to d_vec.
    k_values = np.linspace(10, d_vec, 20, dtype=int)
    errors_k = []
    times_k = []
    for k in k_values:
        start = time.time()
        R = torch.randn(k, d_vec) / np.sqrt(k)
        y_approx = (v_row @ R.t()) @ (R @ M)
        t_k = time.time() - start
        times_k.append(t_k)
        err_k = torch.norm(y_exact - y_approx, p='fro') / torch.norm(y_exact, p='fro')
        errors_k.append(err_k.item())
    error_dict_vec[d_vec] = (k_values, errors_k)
    time_dict_vec[d_vec] = (k_values, times_k)
    print(f"Vector size {d_vec}: k from {k_values[0]} to {k_values[-1]}, final relative error: {errors_k[-1]:.4f}")

# Combined Plot: Relative error vs. projection dimension k for different vector sizes.
plt.figure(figsize=(8, 5))
for d_vec in vector_sizes:
    k_vals, err_vals = error_dict_vec[d_vec]
    plt.plot(k_vals, err_vals, marker='o', linestyle='-', label=f"Vector Size {d_vec}")
plt.xlabel('Projection Dimension k')
plt.ylabel('Relative Frobenius Norm Error')
plt.title('Vector * Matrix: Error vs. k for Different Vector Sizes')
plt.legend(title="Vector Size")
plt.grid(True)
plt.savefig("jl_vec_size_error_vs_k.png")
plt.show()

# Combined Plot: Computation time vs. projection dimension k for different vector sizes.
plt.figure(figsize=(8, 5))
for d_vec in vector_sizes:
    k_vals, time_vals = time_dict_vec[d_vec]
    plt.plot(k_vals, time_vals, marker='o', linestyle='-', label=f"Vector Size {d_vec}")
plt.xlabel('Projection Dimension k')
plt.ylabel('Computation Time (seconds)')
plt.title('Vector * Matrix: Time vs. k for Different Vector Sizes')
plt.legend(title="Vector Size")
plt.grid(True)
plt.savefig("jl_vec_size_time_vs_k.png")
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from sklearn.random_projection import GaussianRandomProjection

def johnson_lindenstrauss_transform(X, target_dim):
    """
    Apply Johnson-Lindenstrauss transform to reduce the dimensionality of X.
    
    Parameters:
        X (numpy.ndarray): Input matrix of shape (n_samples, original_dim).
        target_dim (int): Target reduced dimensionality.
    
    Returns:
        X_reduced (numpy.ndarray): Transformed matrix of shape (n_samples, target_dim).
        P (numpy.ndarray): Random projection matrix.
    """
    n_samples, original_dim = X.shape
    assert target_dim < original_dim, "Target dimension must be smaller than original."

    # Create a random Gaussian projection matrix
    transformer = GaussianRandomProjection(n_components=target_dim)
    X_reduced = transformer.fit_transform(X)
    P = transformer.components_.T  # Extract projection matrix

    return X_reduced, P

def adjust_transformation_matrix(P, W):
    """
    Adjust a transformation matrix W so that applying it after JL transform produces
    approximately the same result as applying the original W before reduction.
    
    Parameters:
        P (numpy.ndarray): Projection matrix used for JL transform.
        W (numpy.ndarray): Original transformation matrix.
    
    Returns:
        W_adjusted (numpy.ndarray): Adjusted transformation matrix.
    """
    P_pseudo_inverse = np.linalg.pinv(P)  # Compute Moore-Penrose pseudo-inverse
    return P_pseudo_inverse @ W  # Adjusted transformation matrix

def compute_outliers(data):
    """
    Compute outliers using Interquartile Range (IQR).
    
    Parameters:
        data (numpy.ndarray): Input array of values.
    
    Returns:
        outliers (numpy.ndarray): Array of outlier values.
    """
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data < lower_bound) | (data > upper_bound)]

def plot_differences(differences_W1, differences_W2):
    """
    Plot boxplots for absolute differences.
    
    Parameters:
        differences_W1 (numpy.ndarray): Differences for single vector output.
        differences_W2 (numpy.ndarray): Differences for matrix output.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[differences_W1, differences_W2.flatten()], showfliers=True, palette="coolwarm")
    plt.xticks([0, 1], ["Single Vector Output", "Matrix Output"])
    plt.ylabel("Absolute Difference")
    plt.title("Comparison of Differences for JL-Transformed Outputs")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Set parameters
    n_samples = 5
    original_dim = 8
    target_dim = 3
    output_dim = 4

    # Generate sample data (n_samples x original_dim)
    np.random.seed(42)
    X = np.random.randn(n_samples, original_dim)

    # Apply JL Transform
    X_reduced, P = johnson_lindenstrauss_transform(X, target_dim)

    # Generate transformation matrices
    W1 = np.random.randn(original_dim, 1)  # Results in a single vector
    W2 = np.random.randn(original_dim, output_dim)  # Results in a transformed matrix

    # Adjust transformation matrices for JL-transformed data
    W1_adjusted = adjust_transformation_matrix(P, W1)
    W2_adjusted = adjust_transformation_matrix(P, W2)

    # Compute original and reduced outputs
    original_output_W1 = X @ W1
    reduced_output_W1 = X_reduced @ W1_adjusted

    original_output_W2 = X @ W2
    reduced_output_W2 = X_reduced @ W2_adjusted

    # Compute absolute differences
    differences_W1 = np.abs(original_output_W1 - reduced_output_W1).flatten()
    differences_W2 = np.abs(original_output_W2 - reduced_output_W2)

    # Compute mean difference per value
    mean_diff_W1 = np.mean(differences_W1)
    mean_diff_W2 = np.mean(differences_W2)

    # Compute outliers
    outliers_W1 = compute_outliers(differences_W1)
    outliers_W2 = compute_outliers(differences_W2.flatten())

    # Display numerical results
    df_mean_differences = pd.DataFrame({
        "Transformation Type": ["Single Vector Output", "Matrix Output"],
        "Mean Absolute Difference": [mean_diff_W1, mean_diff_W2]
    })

    df_outliers_W1 = pd.DataFrame({"Outliers (Single Vector Output)": outliers_W1})
    df_outliers_W2 = pd.DataFrame({"Outliers (Matrix Output)": outliers_W2})

    # Plot the differences
    plot_differences(differences_W1, differences_W2)

    # Print mean absolute differences and outliers
    print("\nMean Absolute Differences:")
    print(df_mean_differences)

    if not df_outliers_W1.empty:
        print("\nOutliers for Single Vector Output:")
        print(df_outliers_W1)
    else:
        print("\nNo outliers found for Single Vector Output.")

    if not df_outliers_W2.empty:
        print("\nOutliers for Matrix Output:")
        print(df_outliers_W2)
    else:
        print("\nNo outliers found for Matrix Output.")


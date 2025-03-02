#!/usr/bin/env python3

import argparse
import math

def a_of_D(D):
    """
    Returns the 'a' parameter for the log model given dimension D.
    a(D) = -16 * (D ^ -0.489)
    """
    return -16.0 * (D ** -0.489)

def b_of_D(D):
    """
    Returns the 'b' parameter for the log model given dimension D.
    b(D) = 90 - 72.5 * (D ^ -0.517)
    """
    return 90.0 - 72.5 * (D ** -0.517)

def predicted_angle(D, x):
    """
    Returns the predicted angle given the dimension D and the x-value,
    using: angle(x) = a(D)*log1p(x) + b(D).
    """
    a_val = a_of_D(D)
    b_val = b_of_D(D)
    return a_val * math.log1p(x) + b_val

def main():
    parser = argparse.ArgumentParser(description="Predict angle based on dimension and x using discovered log-fit formulas.")
    parser.add_argument("--dim", type=int, required=True, help="Number of dimensions (D).")
    parser.add_argument("--x", type=float, required=True, help="X-value (e.g., # of vectors).")

    args = parser.parse_args()

    # Calculate the predicted angle
    angle = predicted_angle(args.dim, args.x)
    
    # Print the result
    print(f"Predicted angle for D={args.dim} and x={args.x} is approximately {angle:.4f} degrees.")

if __name__ == "__main__":
    main()


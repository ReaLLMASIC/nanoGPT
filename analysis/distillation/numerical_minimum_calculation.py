import argparse
from mpmath import mp, mpf, pi, sqrt, sin, gamma, quad
from tqdm import tqdm
import sys

def h_density_mp(t, p):
    """
    The probability density function using mpmath.
    """
    t = mpf(t)
    p = mpf(p)
    if mpf(0) <= t <= pi:
        term1 = mpf(1) / sqrt(pi)
        term2 = gamma(p / 2) / gamma((p - 1) / 2)
        term3 = (sin(t))**(p - 2)
        return term1 * term2 * term3
    return mpf(0)

def survival_function_single_angle_mp(theta, p):
    """
    The survival function using mpmath.
    """
    theta = mpf(theta)
    p = mpf(p)
    if theta < 0:
        return mpf(1.0)
    if theta > pi:
        return mpf(0.0)
    integral = quad(lambda t: h_density_mp(t, p), [theta, pi])
    return integral

def survival_function_min_angle_mp(theta, n, p):
    """
    The survival function of the minimum angle using mpmath.
    """
    theta = mpf(theta)
    n = mpf(n)
    p = mpf(p)
    if theta < 0:
        return mpf(1.0)
    if theta > pi:
        return mpf(0.0)
    return survival_function_single_angle_mp(theta, p)**n

def expected_minimum_angle_numerical_mp(n, p, dps=50, progress_bar=True):
    """
    Calculates the expected minimum angle numerically using mpmath.
    """
    if p < 2:
        raise ValueError("Dimension p must be >= 2")
    if n < 1:
        raise ValueError("Number of prior vectors n must be >= 1")

    mp.dps = dps
    n_mp = mpf(n)
    p_mp = mpf(p)

    # Dummy integration to estimate the work for the progress bar
    try:
        quad(lambda t: survival_function_min_angle_mp(t, n_mp, p_mp), [mpf(0), pi], maxdegree=1)
        total_intervals = 2 # Initial guess
    except Exception:
        total_intervals = 200 # Fallback if dummy integration fails

    progress_bar_active = progress_bar and sys.stderr.isatty()

    if progress_bar_active:
        pbar = tqdm(total=total_intervals, desc="Integrating", unit="interval")

    def integrand_with_progress(t):
        if progress_bar_active:
            # mpmath's quad doesn't directly expose progress,
            # so we can't update precisely. We'll just update
            # a small amount per call as a heuristic.
            pbar.update(0)
        return survival_function_min_angle_mp(t, n_mp, p_mp)

    # We need to manually manage the integration intervals for better progress
    # This is a simplified approach and might not be perfectly accurate
    num_segments = total_intervals
    step = pi / num_segments
    expected_value_rad = mpf(0)

    for i in range(num_segments):
        lower = mpf(i) * step
        upper = mpf(i + 1) * step
        try:
            segment_integral = quad(integrand_with_progress, [lower, upper])
            expected_value_rad += segment_integral
            if progress_bar_active:
                pbar.update(1)
        except Exception as e:
            if progress_bar_active:
                pbar.set_postfix({"error": str(e)}, refresh=True)
            # Re-raise the exception to be caught outside
            raise

    if progress_bar_active:
        pbar.close()

    return expected_value_rad

def expected_minimum_angle_approx_mp(n, p, dps=50):
    """
    Calculates the approximate expected minimum angle for large n using mpmath.
    """
    if p < 2:
        raise ValueError("Dimension p must be >= 2")
    if n < 1:
        raise ValueError("Number of prior vectors n must be >= 1")

    mp.dps = dps
    n_mp = mpf(n)
    p_mp = mpf(p)

    if p_mp == 2:
        return (mpf(1) / n_mp) * gamma(mpf(1))
    if p_mp > 2:
        K = (mpf(1) / (mpf(4) * sqrt(pi))) * gamma(p_mp / 2) / gamma((p_mp + 1) / 2)
        term1 = mpf(1) / (p_mp - 1)
        term2 = (n_mp * K)**(-mpf(1) / (p_mp - 1))
        term3 = gamma(mpf(1) / (p_mp - 1))
        return term1 * term2 * term3
    return mpf('nan') # Should not reach here

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the expected minimum angle using arbitrary precision.")
    parser.add_argument("-p", "--dimension", type=int, required=True, help="Dimension of the space (p >= 2)")
    parser.add_argument("-n", "--num_prior_vectors", type=int, required=True, help="Number of prior random vectors (n >= 1)")
    parser.add_argument("-a", "--approximate", action="store_true", help="Use the large n approximation")
    parser.add_argument("-dps", "--precision", type=int, default=50, help="Decimal places of precision for mpmath (default: 50)")
    parser.add_argument("--no_progress", action="store_true", help="Disable the progress bar")

    args = parser.parse_args()
    dimension = args.dimension
    num_prior = args.num_prior_vectors
    use_approx = args.approximate
    precision_dps = args.precision
    disable_progress = args.no_progress

    if dimension < 2:
        print("Error: Dimension p must be >= 2")
        exit(1)
    if num_prior < 1:
        print("Error: Number of prior vectors n must be >= 1")
        exit(1)

    mp.dps = precision_dps

    if use_approx:
        try:
            expected_angle_rad = expected_minimum_angle_approx_mp(num_prior, dimension, dps=precision_dps)
            expected_angle_deg = expected_angle_rad * (mpf(180) / pi)
            print(f"Approximate expected minimum angle (n={num_prior}, p={dimension}, dps={precision_dps}): {expected_angle_deg} degrees")
        except ValueError as e:
            print(f"Error: {e}")
    else:
        try:
            expected_angle_rad = expected_minimum_angle_numerical_mp(num_prior, dimension, dps=precision_dps, progress_bar=not disable_progress)
            expected_angle_deg = expected_angle_rad * (mpf(180) / pi)
            print(f"\nNumerically calculated expected minimum angle (n={num_prior}, p={dimension}, dps={precision_dps}): {expected_angle_deg} degrees")
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"\nAn error occurred during numerical integration: {e}")

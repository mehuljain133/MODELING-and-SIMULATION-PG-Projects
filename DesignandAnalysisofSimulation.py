# Unit-III Design and Analysis of simulation experiments: Data collection, Identifying distributions with data, Parameter estimation, Goodness of fit tests, Selecting input models without data, Multivariate and time series input models, Verification and validation of models, Steady-state simulation, Terminating simulation, Confidence interval estimation, Output analysis for steady state simulation, Stochastic simulation.

"""
Modeling and Simulation - Unit III: Design and Analysis of Simulation Experiments
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# --------------------------
# 1. Data Collection (simulated data)
# --------------------------

def generate_sample_data(dist='normal', size=1000):
    if dist == 'normal':
        data = np.random.normal(loc=50, scale=5, size=size)
    elif dist == 'exponential':
        data = np.random.exponential(scale=10, size=size)
    else:
        data = np.random.uniform(low=0, high=100, size=size)
    return data

# --------------------------
# 2. Identifying Distributions with Data & Parameter Estimation
# --------------------------

def fit_distribution(data):
    """
    Fit data to several distributions and pick best fit by KS test
    """
    distributions = {
        'normal': stats.norm,
        'exponential': stats.expon,
        'uniform': stats.uniform
    }
    best_fit = None
    best_p = 0
    for name, dist in distributions.items():
        params = dist.fit(data)
        D, p = stats.kstest(data, name, args=params)
        print(f"Fit {name}: KS stat={D:.3f}, p-value={p:.3f}")
        if p > best_p:
            best_p = p
            best_fit = (name, params)
    return best_fit

# --------------------------
# 3. Goodness of Fit Tests (Kolmogorov-Smirnov)
# --------------------------

def plot_fit(data, dist_name, params):
    plt.hist(data, bins=30, density=True, alpha=0.5, label='Data')
    x = np.linspace(min(data), max(data), 100)
    if dist_name == 'normal':
        pdf = stats.norm.pdf(x, *params)
    elif dist_name == 'exponential':
        pdf = stats.expon.pdf(x, *params)
    elif dist_name == 'uniform':
        pdf = stats.uniform.pdf(x, *params)
    plt.plot(x, pdf, 'r-', label=f'{dist_name} fit')
    plt.legend()
    plt.show()

# --------------------------
# 4. Selecting Input Models Without Data (Rule-based example)
# --------------------------

def select_input_model():
    """
    Simple heuristic to select input model if no data available.
    For example, if system shows constant rate of occurrence -> exponential.
    """
    # Assume interarrival times known to be memoryless -> Exponential
    print("Selected input model: Exponential (memoryless property assumed)")

# --------------------------
# 5. Multivariate and Time Series Input Models (Simple examples)
# --------------------------

def multivariate_input_model():
    """
    Generate bivariate normal distribution sample data
    """
    mean = [0, 0]
    cov = [[1, 0.8], [0.8, 1]]  # covariance matrix with correlation
    data = np.random.multivariate_normal(mean, cov, 500)
    plt.scatter(data[:,0], data[:,1], alpha=0.5)
    plt.title("Bivariate Normal Sample")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def time_series_input_model():
    """
    Simulate AR(1) time series: X_t = 0.7*X_{t-1} + noise
    """
    n = 500
    X = np.zeros(n)
    noise = np.random.normal(0,1,n)
    for t in range(1, n):
        X[t] = 0.7 * X[t-1] + noise[t]
    plt.plot(X)
    plt.title("Simulated AR(1) Time Series")
    plt.show()

# --------------------------
# 6. Verification and Validation of Models
# --------------------------

def verify_model(output_data):
    """
    Simple verification: check for logical errors (non-negative output etc)
    """
    if np.any(output_data < 0):
        print("Verification failed: negative values found")
    else:
        print("Verification passed: no negative values")

def validate_model(simulated_data, real_data):
    """
    Validation: Compare simulated output with real data statistically
    """
    D, p = stats.ks_2samp(simulated_data, real_data)
    print(f"Validation KS test p-value: {p:.3f}")
    if p > 0.05:
        print("Model validated: simulated data matches real data distribution")
    else:
        print("Model validation failed")

# --------------------------
# 7. Steady-State and Terminating Simulation
# --------------------------

def terminating_simulation():
    print("Terminating simulation: run for fixed time/replications")
    # Example: sum of 100 random variables
    data = np.random.poisson(lam=5, size=100)
    print(f"Terminating simulation output mean: {np.mean(data):.2f}")

def steady_state_simulation():
    print("Steady-state simulation: run until system stabilizes")
    # Example: running average converges
    data = np.random.exponential(scale=10, size=1000)
    running_mean = np.cumsum(data) / np.arange(1, len(data)+1)
    plt.plot(running_mean)
    plt.title("Running mean converging (steady state)")
    plt.show()

# --------------------------
# 8. Confidence Interval Estimation
# --------------------------

def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2., n-1)
    print(f"{int(confidence*100)}% confidence interval: ({mean-h:.3f}, {mean+h:.3f})")

# --------------------------
# 9. Output Analysis for Steady State Simulation
# --------------------------

def output_analysis_steady_state(data):
    """
    Example: check mean and confidence interval assuming steady state reached
    """
    print(f"Mean output: {np.mean(data):.3f}")
    confidence_interval(data)

# --------------------------
# 10. Stochastic Simulation (Monte Carlo example)
# --------------------------

def monte_carlo_integration(func, a, b, n=10000):
    """Estimate integral of func from a to b using Monte Carlo"""
    x = np.random.uniform(a, b, n)
    y = func(x)
    integral = (b - a) * np.mean(y)
    return integral

def demo_monte_carlo():
    # Integral of sin(x) from 0 to pi = 2
    result = monte_carlo_integration(np.sin, 0, np.pi, n=10000)
    print(f"Monte Carlo integral estimate of sin(x) from 0 to pi: {result:.4f}")

# --------------------------
# Main demo
# --------------------------

if __name__ == "__main__":
    print("1. Data Collection:")
    data = generate_sample_data('normal')
    print(f"Sample data mean: {np.mean(data):.2f}, std dev: {np.std(data):.2f}")

    print("\n2. Identify distribution and parameter estimation:")
    best_fit = fit_distribution(data)
    print(f"Best fit distribution: {best_fit[0]}, Parameters: {best_fit[1]}")
    plot_fit(data, best_fit[0], best_fit[1])

    print("\n3. Selecting Input Model Without Data:")
    select_input_model()

    print("\n4. Multivariate Input Model Demo:")
    multivariate_input_model()

    print("\n5. Time Series Input Model Demo:")
    time_series_input_model()

    print("\n6. Verification and Validation of Models:")
    verify_model(data)
    # For validation, simulate another dataset with same dist
    simulated_data = generate_sample_data('normal')
    validate_model(data, simulated_data)

    print("\n7. Terminating Simulation:")
    terminating_simulation()

    print("\n8. Steady-State Simulation:")
    steady_state_simulation()

    print("\n9. Confidence Interval Estimation:")
    confidence_interval(data)

    print("\n10. Output Analysis for Steady-State Simulation:")
    output_analysis_steady_state(data)

    print("\n11. Stochastic Simulation - Monte Carlo Integration:")
    demo_monte_carlo()

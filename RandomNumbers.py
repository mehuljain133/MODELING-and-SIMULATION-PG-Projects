# Unit-II Random Numbers: True and pseudo random numbers, Properties of random numbers, Generation of pseudo random numbers, Tests for randomness, Random variate generation using inverse transformation, Direct transformation, Convolution method and Acceptance-rejection method.

"""
Modeling and Simulation - Unit II: Random Numbers and Random Variate Generation
"""

import random
import math
import numpy as np
from scipy.stats import chisquare

# --------------------------
# 1. True vs Pseudo Random Numbers
# --------------------------

def true_random_demo():
    """
    True random numbers come from physical phenomena (e.g., radioactive decay).
    Here, we simulate it conceptually using os.urandom for entropy source.
    """
    import os
    print("True random bytes:", os.urandom(5))

def pseudo_random_demo(seed=123):
    """
    Pseudo random numbers generated algorithmically (deterministic).
    Python's random module is pseudo random.
    """
    random.seed(seed)
    print("Pseudo random numbers:")
    for _ in range(5):
        print(random.random())

# --------------------------
# 2. Properties of Random Numbers
# --------------------------

def check_uniformity(samples=1000):
    """Check uniformity by plotting histogram"""
    data = [random.random() for _ in range(samples)]
    import matplotlib.pyplot as plt
    plt.hist(data, bins=10, edgecolor='black')
    plt.title('Uniformity Check: Histogram of Random Numbers')
    plt.show()

def check_independence(samples=1000):
    """
    A simple independence test: plot random numbers vs their lagged values.
    """
    import matplotlib.pyplot as plt
    data = [random.random() for _ in range(samples)]
    plt.scatter(data[:-1], data[1:], alpha=0.5)
    plt.title('Independence Check: Scatter plot of consecutive random numbers')
    plt.xlabel('X_i')
    plt.ylabel('X_(i+1)')
    plt.show()

# --------------------------
# 3. Generation of Pseudo Random Numbers - Linear Congruential Generator (LCG)
# --------------------------

class LCG:
    def __init__(self, seed=1, a=1664525, c=1013904223, m=2**32):
        self.seed = seed
        self.a = a
        self.c = c
        self.m = m
        self.state = seed

    def next(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m

def demo_lcg():
    lcg = LCG(seed=42)
    print("LCG generated pseudo random numbers:")
    for _ in range(5):
        print(lcg.next())

# --------------------------
# 4. Tests for Randomness - Chi-Square Test for Uniformity
# --------------------------

def chi_square_test_uniform(samples=1000, bins=10):
    data = [random.random() for _ in range(samples)]
    counts, _ = np.histogram(data, bins=bins)
    expected = [samples / bins] * bins
    chi2_stat, p_value = chisquare(counts, f_exp=expected)
    print(f"Chi-square statistic: {chi2_stat:.2f}, p-value: {p_value:.3f}")
    if p_value > 0.05:
        print("Fail to reject null hypothesis: Data is uniform")
    else:
        print("Reject null hypothesis: Data is not uniform")

# --------------------------
# 5. Random Variate Generation Methods
# --------------------------

# 5.1 Inverse Transformation Method
def inverse_transform_exponential(lambd=1.0):
    """Generate an exponential random variate using inverse transform method"""
    u = random.random()
    x = -math.log(1 - u) / lambd
    return x

def demo_inverse_transform():
    samples = [inverse_transform_exponential(0.5) for _ in range(5)]
    print("Exponential random variates by inverse transform:")
    print(samples)

# 5.2 Direct Transformation Method (Box-Muller for normal distribution)
def box_muller():
    u1, u2 = random.random(), random.random()
    z1 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    z2 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
    return z1, z2

def demo_box_muller():
    print("Normal random variates using Box-Muller:")
    for _ in range(3):
        z1, z2 = box_muller()
        print(f"{z1:.4f}, {z2:.4f}")

# 5.3 Convolution Method (sum of uniform variables approximates normal)
def convolution_method(n=12):
    """Generate approximately normal using sum of uniforms"""
    total = sum(random.random() for _ in range(n))
    return total - n/2

def demo_convolution():
    samples = [convolution_method() for _ in range(5)]
    print("Normal approx variates by convolution method:")
    print(samples)

# 5.4 Acceptance-Rejection Method
def acceptance_rejection():
    """
    Generate random variates from f(x) = 2x, 0 <= x <=1 (triangular pdf)
    using acceptance rejection method with uniform proposal.
    """
    while True:
        x = random.random()  # proposal from uniform(0,1)
        u = random.random()
        if u <= 2 * x:  # acceptance criterion
            return x

def demo_acceptance_rejection():
    samples = [acceptance_rejection() for _ in range(5)]
    print("Random variates by acceptance-rejection method (f(x) = 2x):")
    print(samples)

# --------------------------
# Main demo
# --------------------------

if __name__ == "__main__":
    print("1. True Random Number Demo:")
    true_random_demo()

    print("\n2. Pseudo Random Number Demo:")
    pseudo_random_demo()

    print("\n3. Properties of Random Numbers - Uniformity Check:")
    check_uniformity()

    print("\n4. Properties of Random Numbers - Independence Check:")
    check_independence()

    print("\n5. Pseudo Random Number Generator - LCG:")
    demo_lcg()

    print("\n6. Chi-Square Test for Uniformity:")
    chi_square_test_uniform()

    print("\n7. Random Variate Generation - Inverse Transform Method:")
    demo_inverse_transform()

    print("\n8. Random Variate Generation - Direct Transformation (Box-Muller):")
    demo_box_muller()

    print("\n9. Random Variate Generation - Convolution Method:")
    demo_convolution()

    print("\n10. Random Variate Generation - Acceptance-Rejection Method:")
    demo_acceptance_rejection()

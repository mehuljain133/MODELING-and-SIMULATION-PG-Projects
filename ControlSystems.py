# Unit-IV Control Systems: Laplace transform, Transfer functions, State- space models, Order of systems, z-transform, Feedback systems, Stability, Observability, Controllability. 

"""
Modeling and Simulation - Unit IV: Control Systems
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, step, lti, dlti, dstep, ss2tf
from scipy.signal import ss2tf, lsim
from scipy.linalg import eigvals
import sympy as sp

# --------------------------
# 1. Laplace Transform (symbolic)
# --------------------------

def laplace_transform_example():
    t, s = sp.symbols('t s')
    f = sp.exp(-2*t) * sp.sin(3*t)
    F = sp.laplace_transform(f, t, s)
    print("Laplace Transform of exp(-2t)*sin(3t):")
    sp.pprint(F[0])

# --------------------------
# 2. Transfer Functions
# --------------------------

def transfer_function_demo():
    # Define numerator and denominator coefficients of TF: H(s) = (s+3) / (s^2 + 4s + 5)
    num = [1, 3]
    den = [1, 4, 5]
    sys = TransferFunction(num, den)
    print(f"Transfer Function: {sys}")
    # Step response plot
    t, y = step(sys)
    plt.plot(t, y)
    plt.title("Step Response of Transfer Function")
    plt.xlabel("Time (s)")
    plt.ylabel("Output")
    plt.grid(True)
    plt.show()

# --------------------------
# 3. State-Space Models
# --------------------------

def state_space_demo():
    # Example system matrices (2nd order system)
    A = np.array([[0, 1], [-5, -4]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    D = np.array([[0]])

    # Create LTI state-space system
    sys = lti(A, B, C, D)

    # Step response simulation
    t = np.linspace(0, 5, 100)
    t, y = step(sys, T=t)
    plt.plot(t, y)
    plt.title("Step Response - State Space Model")
    plt.xlabel("Time (s)")
    plt.ylabel("Output")
    plt.grid(True)
    plt.show()

    # Convert state-space to transfer function
    num, den = ss2tf(A, B, C, D)
    print("Transfer Function numerator coefficients:", num)
    print("Transfer Function denominator coefficients:", den)

# --------------------------
# 4. Order of System
# --------------------------

def system_order(num, den):
    order = len(den) - 1
    print(f"Order of system (from denominator polynomial): {order}")
    return order

# --------------------------
# 5. Z-Transform (Discrete-time)
# --------------------------

def z_transform_example():
    n, z = sp.symbols('n z')
    f = sp.Function('f')(n)
    # Example: f(n) = (0.5)**n * u(n) (unit step)
    f = 0.5**n
    Fz = sp.summation(f * z**(-n), (n, 0, sp.oo))
    print("Z-Transform of (0.5)^n * u(n):")
    sp.pprint(Fz)

# --------------------------
# 6. Feedback Systems
# --------------------------

def feedback_system_demo():
    # Open loop TF: G(s) = 1 / (s(s+2))
    G_num = [1]
    G_den = [1, 2, 0]
    G = TransferFunction(G_num, G_den)

    # Feedback H(s) = 1 (unity feedback)
    H = TransferFunction([1], [1])

    # Closed loop transfer function: T = G / (1 + G*H)
    num_cl = np.polymul(G_num, H.den)
    den_cl = np.polyadd(np.polymul(G_den, H.den), np.polymul(G_num, H.num))
    T = TransferFunction(num_cl, den_cl)

    print(f"Closed-loop Transfer Function:\nNumerator: {T.num}\nDenominator: {T.den}")

    # Step response of closed-loop system
    t, y = step(T)
    plt.plot(t, y)
    plt.title("Closed-Loop Step Response")
    plt.xlabel("Time (s)")
    plt.ylabel("Output")
    plt.grid(True)
    plt.show()

# --------------------------
# 7. Stability (Poles of system)
# --------------------------

def check_stability(den):
    poles = np.roots(den)
    print(f"Poles of system: {poles}")
    stable = all(np.real(p) < 0 for p in poles)
    print(f"System is {'stable' if stable else 'unstable'}")

# --------------------------
# 8. Observability and Controllability
# --------------------------

def observability_controllability(A, B, C):
    n = A.shape[0]

    # Controllability matrix
    ctrb = B
    for i in range(1, n):
        ctrb = np.hstack((ctrb, np.linalg.matrix_power(A, i) @ B))

    # Observability matrix
    obsv = C
    for i in range(1, n):
        obsv = np.vstack((obsv, C @ np.linalg.matrix_power(A, i)))

    ctrb_rank = np.linalg.matrix_rank(ctrb)
    obsv_rank = np.linalg.matrix_rank(obsv)

    print(f"Controllability matrix rank: {ctrb_rank} (system order: {n})")
    print(f"Observability matrix rank: {obsv_rank} (system order: {n})")

    print(f"System is {'controllable' if ctrb_rank == n else 'not controllable'}")
    print(f"System is {'observable' if obsv_rank == n else 'not observable'}")

# --------------------------
# Main demo
# --------------------------

if __name__ == "__main__":
    print("1. Laplace Transform Example:")
    laplace_transform_example()

    print("\n2. Transfer Function Demo:")
    transfer_function_demo()

    print("\n3. State-Space Model Demo:")
    state_space_demo()

    print("\n4. System Order:")
    # Example denominator polynomial for TF: s^3 + 4s^2 + 5s + 6
    system_order([1, 3], [1, 4, 5, 6])

    print("\n5. Z-Transform Example:")
    z_transform_example()

    print("\n6. Feedback System Demo:")
    feedback_system_demo()

    print("\n7. Stability Check:")
    check_stability([1, 4, 5, 6])

    print("\n8. Observability and Controllability:")
    A = np.array([[0, 1], [-5, -4]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    observability_controllability(A, B, C)

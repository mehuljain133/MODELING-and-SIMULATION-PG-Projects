# Unit-V Statistical Models in Simulation: Common discrete and continuous distributions, Poisson process, Markov chain, Empirical distributions, Queuing systems, Transient and steady-state behavior, performance, Network of queues.

"""
Modeling and Simulation - Unit V: Statistical Models in Simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# --------------------------
# 1. Common Discrete and Continuous Distributions
# --------------------------

def sample_distributions():
    np.random.seed(0)
    # Discrete: Binomial (n=10, p=0.5)
    binomial_sample = np.random.binomial(n=10, p=0.5, size=1000)

    # Continuous: Normal (mean=0, std=1)
    normal_sample = np.random.normal(loc=0, scale=1, size=1000)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.hist(binomial_sample, bins=range(12), alpha=0.7, color='skyblue', edgecolor='black')
    plt.title("Binomial Distribution (n=10, p=0.5)")
    plt.xlabel("Number of successes")
    plt.ylabel("Frequency")

    plt.subplot(1,2,2)
    plt.hist(normal_sample, bins=30, alpha=0.7, color='salmon', edgecolor='black')
    plt.title("Normal Distribution (mean=0, std=1)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# --------------------------
# 2. Poisson Process Simulation
# --------------------------

def simulate_poisson_process(rate=2, time=10):
    """
    Simulate Poisson process arrivals using exponential inter-arrival times
    """
    arrival_times = []
    t = 0
    while t < time:
        t += np.random.exponential(1/rate)
        if t < time:
            arrival_times.append(t)
    print(f"Simulated {len(arrival_times)} arrivals in {time} time units with rate {rate}")
    plt.eventplot(arrival_times, colors='b')
    plt.title("Poisson Process Arrival Times")
    plt.xlabel("Time")
    plt.show()
    return arrival_times

# --------------------------
# 3. Markov Chain Simulation
# --------------------------

def simulate_markov_chain(transition_matrix, states, start_state, steps=20):
    current_state = start_state
    path = [current_state]

    for _ in range(steps):
        current_index = states.index(current_state)
        next_state = np.random.choice(states, p=transition_matrix[current_index])
        path.append(next_state)
        current_state = next_state
    
    print("Markov chain path:", path)
    plt.plot(path, marker='o')
    plt.title("Markov Chain State Transitions")
    plt.xlabel("Step")
    plt.ylabel("State")
    plt.grid(True)
    plt.show()

# --------------------------
# 4. Empirical Distribution
# --------------------------

def empirical_distribution(data):
    counts = Counter(data)
    total = len(data)
    x = np.array(list(counts.keys()))
    y = np.array([counts[k]/total for k in x])

    plt.bar(x, y, color='lightgreen', edgecolor='black')
    plt.title("Empirical Distribution")
    plt.xlabel("Value")
    plt.ylabel("Probability")
    plt.show()

# --------------------------
# 5. Simple M/M/1 Queuing System Simulation
# --------------------------

def simulate_mm1_queue(arrival_rate=1.5, service_rate=2.0, sim_time=50):
    np.random.seed(0)
    arrival_times = []
    service_times = []

    # Generate arrival times (Poisson process)
    t = 0
    while t < sim_time:
        t += np.random.exponential(1/arrival_rate)
        if t < sim_time:
            arrival_times.append(t)

    # Service times (exponential)
    service_times = np.random.exponential(1/service_rate, len(arrival_times))

    queue = []
    departure_times = []
    current_time = 0

    for i, arrival in enumerate(arrival_times):
        if current_time < arrival:
            current_time = arrival
        start_service = current_time
        end_service = start_service + service_times[i]
        departure_times.append(end_service)
        current_time = end_service

    wait_times = np.array(departure_times) - np.array(arrival_times)
    print(f"Average waiting time in system: {np.mean(wait_times):.3f}")

    plt.hist(wait_times, bins=30, color='orange', edgecolor='black')
    plt.title("Waiting Time Distribution in M/M/1 Queue")
    plt.xlabel("Waiting Time")
    plt.ylabel("Frequency")
    plt.show()

# --------------------------
# 6. Transient and Steady-State Behavior
# --------------------------

def transient_steady_state_analysis(wait_times):
    """
    Plot cumulative average waiting time to show convergence (steady state)
    """
    cum_avg = np.cumsum(wait_times) / np.arange(1, len(wait_times)+1)
    plt.plot(cum_avg)
    plt.title("Transient to Steady-State Behavior (Average Waiting Time)")
    plt.xlabel("Number of Customers")
    plt.ylabel("Average Waiting Time")
    plt.grid(True)
    plt.show()

# --------------------------
# 7. Network of Queues (Tandem Queues) Simulation
# --------------------------

def network_of_queues():
    """
    Simulate two M/M/1 queues in series (tandem queue)
    """
    np.random.seed(1)
    sim_time = 50
    arrival_rate = 1.0
    service_rate_1 = 1.5
    service_rate_2 = 2.0

    # Arrival times to first queue
    t = 0
    arrivals_q1 = []
    while t < sim_time:
        t += np.random.exponential(1/arrival_rate)
        if t < sim_time:
            arrivals_q1.append(t)
    arrivals_q1 = np.array(arrivals_q1)

    # Service times first queue
    service_q1 = np.random.exponential(1/service_rate_1, len(arrivals_q1))

    # Departure times first queue
    dep_q1 = []
    current_time = 0
    for i, arrival in enumerate(arrivals_q1):
        if current_time < arrival:
            current_time = arrival
        current_time += service_q1[i]
        dep_q1.append(current_time)
    dep_q1 = np.array(dep_q1)

    # Arrival times second queue = departure from first queue
    arrivals_q2 = dep_q1
    service_q2 = np.random.exponential(1/service_rate_2, len(arrivals_q2))

    # Departure times second queue
    dep_q2 = []
    current_time = 0
    for i, arrival in enumerate(arrivals_q2):
        if current_time < arrival:
            current_time = arrival
        current_time += service_q2[i]
        dep_q2.append(current_time)
    dep_q2 = np.array(dep_q2)

    # Calculate total time in system (both queues)
    total_time = dep_q2 - arrivals_q1
    print(f"Average total time in system for tandem queues: {np.mean(total_time):.3f}")

    plt.hist(total_time, bins=30, color='purple', edgecolor='black')
    plt.title("Total Time Distribution in Network of Queues")
    plt.xlabel("Total Time in System")
    plt.ylabel("Frequency")
    plt.show()

# --------------------------
# Main demo
# --------------------------

if __name__ == "__main__":
    print("1. Sample Common Distributions:")
    sample_distributions()

    print("\n2. Poisson Process Simulation:")
    arrivals = simulate_poisson_process(rate=3, time=10)

    print("\n3. Markov Chain Simulation:")
    states = ['Sunny', 'Cloudy', 'Rainy']
    P = [
        [0.7, 0.2, 0.1],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5]
    ]
    simulate_markov_chain(P, states, start_state='Sunny', steps=30)

    print("\n4. Empirical Distribution Example:")
    data = np.random.randint(0, 5, size=1000)
    empirical_distribution(data)

    print("\n5. M/M/1 Queue Simulation:")
    simulate_mm1_queue(arrival_rate=1.5, service_rate=2.0, sim_time=100)

    print("\n6. Transient and Steady-State Analysis:")
    # reuse wait times from M/M/1 simulation for demo
    # We simulate again here to get wait times array:
    np.random.seed(0)
    arrivals = []
    t = 0
    sim_time = 100
    while t < sim_time:
        t += np.random.exponential(1/1.5)
        if t < sim_time:
            arrivals.append(t)
    arrivals = np.array(arrivals)
    service_times = np.random.exponential(1/2.0, len(arrivals))
    dep_times = []
    current_time = 0
    for i, arrival in enumerate(arrivals):
        if current_time < arrival:
            current_time = arrival
        current_time += service_times[i]
        dep_times.append(current_time)
    dep_times = np.array(dep_times)
    wait_times = dep_times - arrivals
    transient_steady_state_analysis(wait_times)

    print("\n7. Network of Queues Simulation:")
    network_of_queues()

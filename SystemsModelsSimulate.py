# Unit-I Systems, Models and Simulation study: Natural and Artificial Systems, Complex Systems, Definition and types of model, Mathematical models, Cyber-physical systems and its modeling, Network models, Steps in simulation study, Advantage and disadvantage of simulation

"""
Modeling and Simulation - Unit I Overview

Topics covered:
- Natural and Artificial Systems
- Complex Systems
- Types of Models
- Mathematical Models
- Cyber-physical Systems
- Network Models
- Steps in Simulation Study
- Advantages and Disadvantages of Simulation
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
import time

# --------------------------
# 1. Natural and Artificial Systems
# --------------------------

class NaturalSystem:
    """Example: A simple ecosystem with prey-predator."""
    def __init__(self, prey=50, predator=10):
        self.prey = prey
        self.predator = predator

    def step(self):
        # Simplified dynamics: prey grows, predator eats prey
        self.prey += int(self.prey * 0.1)  # prey growth
        eaten = min(self.prey, self.predator * 2)
        self.prey -= eaten
        self.predator += eaten // 3 - 1  # predator growth minus death

    def status(self):
        return f"Prey: {self.prey}, Predator: {self.predator}"

class ArtificialSystem:
    """Example: A simple queue system (artificial)."""
    def __init__(self, queue=[]):
        self.queue = queue

    def add_customer(self, customer):
        self.queue.append(customer)

    def serve_customer(self):
        if self.queue:
            return self.queue.pop(0)
        else:
            return None

# --------------------------
# 2. Complex Systems
# --------------------------

class ComplexSystem:
    """A network of interacting agents."""
    def __init__(self, size):
        self.graph = nx.erdos_renyi_graph(size, 0.1)

    def simulate_interaction(self):
        # Randomly choose nodes to interact and influence states
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if neighbors:
                influence = sum(random.choice([0, 1]) for _ in neighbors)
                # For demo, just print interaction influence
                print(f"Node {node} influenced by {influence} neighbors")

# --------------------------
# 3. Definition and Types of Models
# --------------------------

"""
Models:
- Physical Models (e.g., globe)
- Mathematical Models (equations)
- Logical Models (flowcharts)
- Simulation Models (software)
"""

# Example: Mathematical model of population growth (Exponential)
def population_growth(P0, r, t):
    return P0 * (2.71828 ** (r * t))  # P(t) = P0 * e^(rt)

# --------------------------
# 4. Mathematical Models
# --------------------------

# Lotka-Volterra predator-prey equations (simplified numeric example)
def lotka_volterra(prey, predator, alpha=0.1, beta=0.02, delta=0.01, gamma=0.1):
    dprey = alpha * prey - beta * prey * predator
    dpredator = delta * prey * predator - gamma * predator
    prey_next = prey + dprey
    predator_next = predator + dpredator
    return max(prey_next, 0), max(predator_next, 0)

# --------------------------
# 5. Cyber-Physical Systems and Its Modeling
# --------------------------

class CyberPhysicalSystem:
    """Simulate a smart thermostat controlling temperature."""
    def __init__(self, temp=25):
        self.temp = temp  # current temperature
        self.set_point = 22  # desired temperature

    def sensor(self):
        # Sensor reads current temp
        return self.temp

    def controller(self):
        # Controller decides action
        if self.temp > self.set_point:
            return "cool"
        elif self.temp < self.set_point:
            return "heat"
        else:
            return "off"

    def actuator(self, action):
        # Actuator changes environment
        if action == "cool":
            self.temp -= 1
        elif action == "heat":
            self.temp += 1

    def step(self):
        current_temp = self.sensor()
        action = self.controller()
        self.actuator(action)
        print(f"Temp: {self.temp:.1f}°C, Action: {action}")

# --------------------------
# 6. Network Models
# --------------------------

def create_network_model():
    G = nx.Graph()
    # Add nodes representing entities
    G.add_nodes_from(range(1, 6))
    # Add edges representing relationships
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]
    G.add_edges_from(edges)
    return G

def plot_network(G):
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.show()

# --------------------------
# 7. Steps in Simulation Study
# --------------------------

simulation_steps = [
    "Problem Formulation",
    "Setting Objectives",
    "Model Conceptualization",
    "Data Collection",
    "Model Translation (coding)",
    "Verification and Validation",
    "Experimentation",
    "Analysis of Results",
    "Documentation and Reporting"
]

def show_simulation_steps():
    print("Steps in Simulation Study:")
    for i, step in enumerate(simulation_steps, 1):
        print(f"{i}. {step}")

# --------------------------
# 8. Advantages and Disadvantages of Simulation
# --------------------------

advantages = [
    "Allows analysis of complex systems",
    "Can model dynamic and stochastic processes",
    "Safe and cost-effective experimentation",
    "Helps in decision making"
]

disadvantages = [
    "Model may not fully represent reality",
    "Requires accurate data",
    "Computationally intensive for large models",
    "Can be time-consuming"
]

def show_advantages_disadvantages():
    print("Advantages of Simulation:")
    for adv in advantages:
        print(f"- {adv}")
    print("\nDisadvantages of Simulation:")
    for dis in disadvantages:
        print(f"- {dis}")

# --------------------------
# Demo of concepts
# --------------------------

if __name__ == "__main__":
    print("1. Natural System Example:")
    eco = NaturalSystem()
    for _ in range(5):
        eco.step()
        print(eco.status())
    
    print("\n2. Artificial System Example:")
    q = ArtificialSystem()
    q.add_customer("Customer 1")
    q.add_customer("Customer 2")
    print(f"Serving: {q.serve_customer()}")
    print(f"Serving: {q.serve_customer()}")

    print("\n3. Complex System Simulation:")
    cs = ComplexSystem(10)
    cs.simulate_interaction()

    print("\n4. Mathematical Model - Population Growth:")
    print(f"Population after 10 units time: {population_growth(100, 0.05, 10):.2f}")

    print("\n5. Mathematical Model - Lotka Volterra:")
    prey, predator = 40, 9
    for _ in range(5):
        prey, predator = lotka_volterra(prey, predator)
        print(f"Prey: {prey:.2f}, Predator: {predator:.2f}")

    print("\n6. Cyber-Physical System Simulation:")
    cps = CyberPhysicalSystem()
    for _ in range(5):
        cps.step()
        time.sleep(0.5)  # simulate time delay

    print("\n7. Network Model:")
    G = create_network_model()
    plot_network(G)

    print("\n8. Steps in Simulation Study:")
    show_simulation_steps()

    print("\n9. Advantages and Disadvantages of Simulation:")
    show_advantages_disadvantages()

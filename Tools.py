# Unit-VI Modeling and Simulation tools: Open Modelica, Netlogo, Python modules for modeling and simulation, GPSS.

"""
Unit VI: Modeling and Simulation Tools

- Overview of OpenModelica, NetLogo, GPSS (notes)
- Python modules demo: SimPy (discrete-event simulation), PyDy (dynamics), matplotlib, scipy

This script demonstrates modeling and simulation using Python modules.
"""

# --- Notes on tools ---
print("""
OpenModelica:
- Open-source modeling environment for complex physical systems using Modelica language.
- Supports equation-based modeling, multi-domain simulation.
- https://openmodelica.org/

NetLogo:
- Agent-based modeling environment; great for simulating complex systems with many interacting agents.
- GUI-based, but supports scripting via its own language.
- https://ccl.northwestern.edu/netlogo/

GPSS (General Purpose Simulation System):
- Discrete-event simulation language, mostly legacy but still in use in industry.
- Typically requires dedicated GPSS software (e.g., GPSS World).

Python Modules for Modeling and Simulation:
- SimPy: Discrete-event simulation.
- PyDy: Multibody dynamics.
- scipy.signal: Control system simulation.
- matplotlib: Visualization.
""")

# --- Python Simulation Demo ---

import simpy
import numpy as np
import matplotlib.pyplot as plt

# Simple SimPy example: single server queue

def customer(env, name, server, service_time):
    arrival_time = env.now
    print(f"{name} arrives at {arrival_time:.2f}")
    with server.request() as request:
        yield request
        wait = env.now - arrival_time
        print(f"{name} starts service at {env.now:.2f} after waiting {wait:.2f}")
        yield env.timeout(service_time)
        print(f"{name} leaves at {env.now:.2f}")

def setup(env, arrival_rate, service_rate):
    server = simpy.Resource(env, capacity=1)
    i = 0
    while True:
        inter_arrival = np.random.exponential(1 / arrival_rate)
        yield env.timeout(inter_arrival)
        service_time = np.random.exponential(1 / service_rate)
        i += 1
        env.process(customer(env, f'Customer {i}', server, service_time))

# Run simulation
env = simpy.Environment()
arrival_rate = 2  # customers per time unit
service_rate = 3  # customers served per time unit

env.process(setup(env, arrival_rate, service_rate))
env.run(until=10)

print("\nSimPy discrete-event simulation completed.")

# --- Additional brief notes on integration ---

print("""
Integration notes:
- OpenModelica models are created in .mo files and simulated via OpenModelica environment or Python bindings (OMPython).
- NetLogo can be controlled programmatically using pyNetLogo, a Python package.
- GPSS simulations are usually run within GPSS software environments; Python interfaces are uncommon.

You can use these Python modules for rapid prototyping or small to medium scale simulations.
""")

# A Hybrid Reinforcement Learning BRKGA with Local Branching for a Sensor Allocation Problem

This repository contains the implementation for the paper: "A Hybrid Reinforcement Learning BRKGA with Local Branching for a Sensor Allocation Problem"

Before running the code, make sure the following directory structure is set:

A folder named `results/` must exist inside the main project folder (e.g., `BRKGA NWS/`).

Inside `results/`, two CSV files must be created manually:

- `regular.csv`
- `semi.csv`

These files will be used to store the output results from each experiment.

You can create them as empty files (they will be written during execution):

---

## About the BRKGA Framework

This implementation uses the **BRKGA-MP-IPR** Python framework, developed by **ceandrade**, available at:

🔗 [https://ceandrade.github.io/brkga_mp_ipr_python](https://ceandrade.github.io/brkga_mp_ipr_python)

In this repository, we propose a **hybrid methodology** that combines the **Biased Random-Key Genetic Algorithm (BRKGA)** with **Reinforcement Learning (RL)-based refinement** incorporated into its **decoder**, as well as **Local Branching** techniques for efficient sensor placement.


### Key Steps in the Algorithm:

1. **Initialize with Multicentrality Heuristic**: We start by applying a **multicentrality heuristic** to generate an initial solution. This step involves calculating centrality measures for the vertices in the graph and using them to prioritize sensor placement, helping the algorithm to begin with a promising initial configuration.
2. **Create a decoder class**: `sap_decoder.py` implements the `decode()` method that translates a chromosome (a list of real numbers in [0, 1]) into a feasible solution.
3. **Incorporate Reinforcement Learning**: After specific iterations, the current solution is passed to an RL model for refinement, which enhances the sensor placement strategy based on learned patterns.
4. **Apply Local Branching (post-evolution)**: Once the BRKGA has completed its evolutionary process, **Local Branching** is used to further improve the best solutions. This technique is applied after the BRKGA has converged to good solutions, optimizing them by making small, localized changes in the sensor placement.

---
---

## Decoder Overview

### SAP-Decoder

The **SAP-Decoder** is responsible for translating a chromosome (real numbers between [0, 1]) into a valid sensor placement solution. The decoder assigns sensors (`X`, `Y`, and `Z`) based on the adjacency list and the secondary adjacency list of the graph, representing a network of vertices. The placement follows these key steps:

1. **Initial Placement**: Sensors of type `X` are placed on vertices based on the chromosome. Sensors of types `Y` and `Z` are then assigned based on proximity (neighborhood relationships).
2. **Reinforcement Learning**: After certain iterations, the current solution is sent to a **Reinforcement Learning model** for refinement. The RL model uses the current placement solution and its associated cost as an initial state and tries to improve it based on learned strategies. If the RL solution provides a better configuration, it replaces the current solution.

---
## Baseline Model (Exact Comparison)

The file CEC_MLPwarm_BRKGA.py is primarily used for implementing the Local Branching technique, but it can also be adapted to serve as an exact baseline model for comparison. This file employs the BRKGA algorithm with a warm start mechanism to optimize sensor allocation.

The function solve_wsn_optimization_docplex solves the MILP sensor allocation model heuristic_solution, using the following parameters: brkga_solution and local_branching_k

#### Adapting the Model for Exact Comparison:
To use this function as an exact baseline model, follow these steps:

1. **Remove the parameters related to the warm start**.
Exclude parameters that pass a heuristic solution or a BRKGA-generated solution to ensure the model starts from scratch. Without any initial solution, the optimization will proceed without leveraging previous information, providing a pure baseline for comparison.

2. **Remove the parameters related to Local Branching**,
Remove the parameters related to Local Branching (such as the delta value) to disable the post-evolution adjustments. This will prevent the model from making small localized improvements to the solution after the BRKGA process, ensuring that only the BRKGA algorithm is used.

Once these adjustments are made, you will be able to run the algorithm as an exact baseline model without the influence of warm start or Local Branching. This allows for a clear comparison between the baseline model and the full hybrid approach that includes these techniques

By doing this, you can run the algorithm **without the warm start** or **Local Branching** and compare its performance with the full hybrid approach (which includes RL and Local Branching) to understand the impact of these optimizations.

---


## ⚙️ Dependencies

This code requires **Python 3.10** and the following Python packages:

- `time`
- `csv`
- `os`
- `numpy`
- `pandas`
- `matplotlib`
- `networkx`
- `docopt`
- `torch`
- `gym`

# SOC-Doom-with-RL

# Project: Multi-Armed Bandit Algorithm Implementations

## Project Overview

This project is a practical exploration of the multi-armed bandit problem, a classic reinforcement learning scenario that exemplifies the exploration-exploitation trade-off. The goal was to implement and compare several fundamental algorithms designed to maximize cumulative reward over time in the face of uncertainty.

The project was completed over a period of three weeks, with each week focusing on a different foundational skill, culminating in the implementation and analysis of the RL agents.

---

## Weekly Learning Summary

### Week 1: Foundations in Version Control

The first week was dedicated to mastering the fundamentals of version control using **Git**. This was crucial for managing the project's codebase effectively.

**Key Concepts Learned:**
-   Initializing a repository (`git init`).
-   The staging/commit lifecycle (`git add`, `git commit`).
-   Tracking changes and viewing project history (`git status`, `git log`).
-   Working with remote repositories on platforms like GitHub (`git push`, `git pull`).
-   Basic branching and merging strategies to manage different features or versions of the code.

### Week 2: Object-Oriented Programming in Python

The second week focused on building a solid foundation in Python's object-oriented programming (OOP) principles. This knowledge was essential for creating a modular and scalable structure for the bandit agents.

**Key Concepts Learned:**
-   Defining classes using the `class` keyword.
-   Initializing object state with the `__init__` constructor.
-   Creating attributes (instance variables) and methods (functions) for a class.
-   The principle of **inheritance**, which was used to create a base `Agent` class and have specific algorithm implementations inherit from it, promoting code reuse (`super().__init__()`).

### Week 3: Implementing Reinforcement Learning Algorithms

The final week involved diving into reinforcement learning theory and applying the programming skills from Week 2 to implement four distinct algorithms for the multi-armed bandit problem.

**Key Concepts Learned:**
-   **The Exploration-Exploitation Trade-off:** Understood the core challenge of deciding whether to exploit the best-known option or explore other options to potentially find a better one.
-   **Regret Minimization:** Learned that the goal of a good bandit algorithm is to minimize "regret"—the difference between the reward obtained and the reward that could have been obtained by choosing the optimal arm at every step.
-   **Algorithm Implementation:** Successfully implemented the following four agents from scratch:
    1.  **Epsilon-Greedy:** A simple yet effective strategy that exploits the best-known arm most of the time but explores a random arm with a fixed probability, `epsilon`.
    2.  **UCB (Upper Confidence Bound):** A deterministic algorithm based on the principle of "optimism in the face of uncertainty." It adds an exploration bonus to arms with high uncertainty (i.e., those that have been pulled less frequently).
    3.  **KL-UCB (Kullback-Leibler UCB):** A more sophisticated version of UCB that uses the KL-divergence to compute a tighter, more accurate confidence bound, often leading to better performance.
    4.  **Thompson Sampling:** A Bayesian algorithm that maintains a probability distribution for the reward of each arm. It chooses an arm by sampling from these distributions, naturally balancing exploration and exploitation.

---

## How to Run the Experiments

This repository contains the code to run simulations comparing the performance of the implemented bandit agents.

### Dependencies
- Python 3
- NumPy
- Matplotlib

### Running the Code
To run the full suite of experiments (S1 and S2) and see the plots displayed on screen, execute the main script from your terminal:

```bash
python main.py

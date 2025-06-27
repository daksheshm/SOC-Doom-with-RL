import numpy as np
from base import Agent, MultiArmedBandit
import matplotlib.pyplot as plt

# Epsilon for numerical stability to avoid log(0)
EPSILON = 1e-9

class KLUCBAgent(Agent):
    """
    Implements the KL-UCB algorithm for a multi-armed bandit problem
    with Bernoulli rewards.
    """
    reward_memory : np.ndarray # A per arm value of how much reward was gathered
    count_memory : np.ndarray # An array of the number of times an arm is pulled

    def __init__(self, time_horizon, bandit:MultiArmedBandit): 
        """
        Initializes the KL-UCB agent.

        Args:
            time_horizon (int): The total number of time steps.
            bandit (MultiArmedBandit): The bandit environment.
        """
        super().__init__(time_horizon, bandit)
        self.bandit = bandit
        num_arms = len(bandit.arms)
        self.reward_memory = np.zeros(num_arms)
        self.count_memory = np.zeros(num_arms)
        self.time_step = 0

    def _kl_bernoulli(self, p, q):
        """Calculates the KL-divergence between two Bernoulli distributions."""
        p = np.clip(p, EPSILON, 1 - EPSILON)
        q = np.clip(q, EPSILON, 1 - EPSILON)
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    def _solve_for_q(self, p_emp, target):
        """
        Solves for the upper bound q using bisection search.
        Finds q > p_emp such that KL(p_emp, q) = target.
        """
        low = p_emp
        high = 1.0 - EPSILON
        
        # Bisection search for 30 iterations is sufficient for high precision
        for _ in range(30):
            q_mid = (low + high) / 2.0
            kl_val = self._kl_bernoulli(p_emp, q_mid)
            if kl_val < target:
                low = q_mid
            else:
                high = q_mid
        return high

    def give_pull(self):
        """
        Selects an arm to pull based on the KL-UCB algorithm.
        1. For the first k pulls, pull each arm once.
        2. Afterwards, calculate the UCB for each arm by solving the KL-divergence equation.
        3. Pull the arm with the highest UCB.
        """
        self.time_step += 1
        num_arms = len(self.bandit.arms)

        # Initialization phase: pull each arm once
        if self.time_step <= num_arms:
            chosen_arm = self.time_step - 1
            reward = self.bandit.pull(chosen_arm)
            self.reinforce(reward, chosen_arm)
            return

        # Main phase: calculate KL-UCB for each arm
        ucb_values = np.zeros(num_arms)
        
        # Use a slightly more complex log term as suggested in the original paper for better performance
        # c=0 is the standard form, other values can be used.
        c = 0 
        rhs = (np.log(self.time_step) + c * np.log(np.log(self.time_step))) / self.count_memory

        for arm in range(num_arms):
            if self.count_memory[arm] == 0:
                # This arm has not been pulled, it's the most uncertain, give it infinite UCB
                ucb_values[arm] = float('inf')
                continue
            
            p_empirical = self.reward_memory[arm] / self.count_memory[arm]
            target = rhs[arm]
            ucb_values[arm] = self._solve_for_q(p_empirical, target)

        # Choose the arm with the highest UCB
        chosen_arm = np.argmax(ucb_values)
        reward = self.bandit.pull(chosen_arm)
        self.reinforce(reward, chosen_arm)

    def reinforce(self, reward, arm):
        """
        Updates the agent's knowledge based on the reward received.

        Args:
            reward (int): The reward (0 or 1).
            arm (int): The index of the arm pulled.
        """
        self.count_memory[arm] += 1
        self.reward_memory[arm] += reward
        self.rewards.append(reward)
 
    def plot_arm_graph(self):
        """
        Plots a bar chart showing the number of times each arm was pulled.
        """
        counts = self.count_memory
        indices = np.arange(len(counts))

        # Plot the data
        plt.figure(figsize=(12, 6))
        plt.bar(indices, counts, color='seagreen', edgecolor='black')

        # Formatting
        plt.title('Arm Pull Counts (KL-UCB)', fontsize=16)
        plt.xlabel('Arm', fontsize=14)
        plt.ylabel('Pull Count', fontsize=14)
        plt.grid(axis='y', linestyle='-')
        plt.xticks(indices, [f'Arm {i}' for i in indices])

        # Annotate the bars with the count values
        for i, count in enumerate(counts):
            plt.text(i, count + 0.5, str(int(count)), ha='center', va='bottom', fontsize=12, color='black')

        # Tight layout to ensure there's no clipping of labels
        plt.tight_layout()

        # Show plot
        plt.show()


# Code to test
if __name__ == "__main__":
    # Init Bandit
    TIME_HORIZON = 10_000
    bandit = MultiArmedBandit(np.array([0.23,0.55,0.76,0.44]))
    agent = KLUCBAgent(TIME_HORIZON, bandit)

    # Loop
    for i in range(TIME_HORIZON):
        agent.give_pull()

    # Plot curves
    agent.plot_reward_vs_time_curve()
    agent.plot_arm_graph()
    bandit.plot_cumulative_regret()

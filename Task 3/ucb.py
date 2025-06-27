import numpy as np
from base import Agent, MultiArmedBandit
import matplotlib.pyplot as plt


class UCBAgent(Agent):
    """
    Implements the UCB1 algorithm for a multi-armed bandit problem.
    """
    reward_memory : np.ndarray # A per arm value of how much reward was gathered
    count_memory : np.ndarray # An array of the number of times an arm is pulled
    c : float # Exploration parameter

    def __init__(self, time_horizon, bandit:MultiArmedBandit, c=2.0): 
        """
        Initializes the UCB agent.

        Args:
            time_horizon (int): The total number of time steps.
            bandit (MultiArmedBandit): The bandit environment.
            c (float): The exploration parameter, controls the width of the confidence bound.
        """
        super().__init__(time_horizon, bandit)
        self.bandit = bandit
        num_arms = len(bandit.arms)
        self.c = c
        self.reward_memory = np.zeros(num_arms)
        self.count_memory = np.zeros(num_arms)
        self.time_step = 0

    def give_pull(self):
        """
        Selects an arm to pull based on the UCB1 algorithm.
        1. For the first k pulls, pull each arm once.
        2. Afterwards, calculate the UCB for each arm.
        3. Pull the arm with the highest UCB.
        """
        num_arms = len(self.bandit.arms)

        # Initialization phase: if an arm has not been pulled yet, pull it.
        # This ensures we have an initial estimate for every arm.
        if self.time_step < num_arms:
            chosen_arm = self.time_step
        else:
            # Main phase: calculate UCB values for all arms
            empirical_means = self.reward_memory / self.count_memory
            
            # Add the exploration bonus
            exploration_bonus = np.sqrt((self.c * np.log(self.time_step)) / self.count_memory)
            
            ucb_values = empirical_means + exploration_bonus
            
            # Choose the arm with the highest UCB value
            chosen_arm = np.argmax(ucb_values)

        # Pull the chosen arm and reinforce
        reward = self.bandit.pull(chosen_arm)
        self.reinforce(reward, chosen_arm)

    def reinforce(self, reward, arm):
        """
        Updates the agent's knowledge (counts and rewards) based on the result.

        Args:
            reward (int): The reward received (0 or 1).
            arm (int): The index of the arm that was pulled.
        """
        self.count_memory[arm] += 1
        self.reward_memory[arm] += reward
        self.time_step += 1
        self.rewards.append(reward)
 
    def plot_arm_graph(self):
        """
        Plots a bar chart showing the number of times each arm was pulled.
        """
        counts = self.count_memory
        indices = np.arange(len(counts))

        # Plot the data
        plt.figure(figsize=(12, 6))
        plt.bar(indices, counts, color='coral', edgecolor='black')

        # Formatting
        plt.title('Arm Pull Counts (UCB1)', fontsize=16)
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
    # Fill with correct constructor, using the standard exploration parameter c=2
    agent = UCBAgent(TIME_HORIZON, bandit, c=2)

    # Loop
    for i in range(TIME_HORIZON):
        agent.give_pull()

    # Plot curves
    agent.plot_reward_vs_time_curve()
    agent.plot_arm_graph()
    bandit.plot_cumulative_regret()

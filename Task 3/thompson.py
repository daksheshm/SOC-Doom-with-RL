import numpy as np
from base import Agent, MultiArmedBandit
import matplotlib.pyplot as plt


class ThompsonSamplingAgent(Agent):
    # Add fields 

    alphas : np.ndarray # Alpha parameters for the beta distribution of each arm 
    betas : np.ndarray # Beta parameters for the beta distribution of each arm
    count_memory : np.ndarray # Array of the number of times an arm is pulled for plotting
    def __init__(self, time_horizon, bandit:MultiArmedBandit,): 
        # Add fields
        super().__init__(time_horizon, bandit)
        self.bandit = bandit
        num_arms = len(bandit.arms)
        self.alphas = np.ones(num_arms)
        self.betas = np.ones(num_arms)
        self.count_memory = np.zeros(num_arms)
        self.time_step = 0

    def give_pull(self):

        """
        Selects an arm to pull based on Thompson Sampling.
        1. Sample a value from each arm's Beta(alpha, beta) distribution.
        2. Choose the arm with the highest sampled value.
        3. Pull the chosen arm and reinforce the agent's knowledge.
        """
        
        sampled_theta = np.random.beta(self.alphas, self.betas)
        
        chosen_arm = np.argmax(sampled_theta)

        reward = self.bandit.pull(chosen_arm)

        self.reinforce(reward, chosen_arm)

    def reinforce(self, reward, arm):
        
        """
        Updates the agent's knowledge (the Beta distribution parameters) based on the reward.

        Args:
            reward(int) : The reward received (0 for failure, 1 for success)
            arm(int) : The index of the arm that was pulled.
        """

        self.alphas[arm]+=reward
        self.betas[arm] += (1-reward)

        self.count_memory[arm] += 1
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
        plt.bar(indices, counts, color='skyblue', edgecolor='black')

        # Formatting
        plt.title('Counts per Category', fontsize=16)
        plt.xlabel('Arm', fontsize=14)
        plt.ylabel('Pull Count', fontsize=14)
        plt.grid(axis='y', linestyle='-')  # Add grid lines for the y-axis
        plt.xticks(indices, [f'Category {i+1}' for i in indices], rotation=45, ha='right')
        # plt.yticks(np.arange(0, max(counts) + 2, step=2))

        # Annotate the bars with the count values
        for i, count in enumerate(counts):
            plt.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=12, color='black')

        # Tight layout to ensure there's no clipping of labels
        plt.tight_layout()

        # Show plot
        plt.show()
        



# Code to test
if __name__ == "__main__":
    # Init Bandit
    TIME_HORIZON = 10_000
    bandit = MultiArmedBandit(np.array([0.23,0.55,0.76,0.44]))
    agent = ThompsonSamplingAgent(TIME_HORIZON, bandit) ## Fill with correct constructor

    # Loop
    for i in range(TIME_HORIZON):
        agent.give_pull()

    # Plot curves
    agent.plot_reward_vs_time_curve()
    agent.plot_arm_graph()
    bandit.plot_cumulative_regret()

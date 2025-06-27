import numpy as np
import matplotlib.pyplot as plt

# Import from the provided base.py and the agent files
from base import MultiArmedBandit
from epsilon_greedy import EpsilonGreedyAgent
from ucb import UCBAgent
from klucb import KLUCBAgent
from thompson import ThompsonSamplingAgent

def run_s1():
    """
    Runs Experiment Set S1 and displays the resulting plots.
    """
    # --- S1 Configuration ---
    TIME_HORIZON = 30_000
    BANDIT_PROBS = np.array([0.23, 0.55, 0.76, 0.44])
    OPTIMAL_ARM_INDEX = np.argmax(BANDIT_PROBS)
    print("S1: Running a single game for 30,000 time steps.")
    print(f"S1: Bandit arm probabilities are {BANDIT_PROBS}")

    agents_to_test = {
        "Epsilon-Greedy (ε=0.1)": (EpsilonGreedyAgent, {"epsilon": 0.1}),
        "UCB1 (c=2)": (UCBAgent, {"c": 2}),
        "KL-UCB": (KLUCBAgent, {}),
        "Thompson Sampling": (ThompsonSamplingAgent, {}),
    }

    results = {}

    # --- Main Simulation Loop for S1 ---
    for name, (agent_class, params) in agents_to_test.items():
        print(f"  - Simulating agent: {name}")
        bandit = MultiArmedBandit(arms=BANDIT_PROBS)
        
        # Call the agent constructor using the 'time_horizon' keyword.
        # The agent's __init__ should handle passing it to super() as 'time_to_run'.
        agent = agent_class(time_horizon=TIME_HORIZON, bandit=bandit, **params)
        
        for _ in range(TIME_HORIZON):
            agent.give_pull()
        
        results[name] = {
            'regret': bandit.cumulative_regret_array[1:], # Exclude initial 0
            'rewards': agent.rewards,
            'optimal_pulls': agent.count_memory[OPTIMAL_ARM_INDEX]
        }
    
    print("S1: Simulations complete. Displaying plots...")

    # --- Plotting S1 Results ---

    # Plot 1: Cumulative Regret
    plt.figure(figsize=(14, 8))
    for name, data in results.items():
        plt.plot(data['regret'], label=name)
    plt.title('S1: Cumulative Regret Comparison', fontsize=16)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Cumulative Regret', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 2: Average Reward
    plt.figure(figsize=(14, 8))
    for name, data in results.items():
        avg_rewards = np.cumsum(data['rewards']) / (np.arange(TIME_HORIZON) + 1)
        plt.plot(avg_rewards, label=name)
    plt.axhline(y=np.max(BANDIT_PROBS), color='black', linestyle='--', label=f'Optimal Reward ({np.max(BANDIT_PROBS)})')
    plt.title('S1: Average Reward Comparison', fontsize=16)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.0)
    plt.show()

    # Plot 3: Optimal Arm Pulls
    plt.figure(figsize=(12, 7))
    agent_names = list(results.keys())
    optimal_pull_counts = [data['optimal_pulls'] for data in results.values()]
    bars = plt.bar(agent_names, optimal_pull_counts, color=['skyblue', 'coral', 'seagreen', 'mediumpurple'])
    plt.title('S1: Number of Times Optimal Arm Was Pulled', fontsize=16)
    plt.xlabel('Agent', fontsize=12)
    plt.ylabel('Pull Count of Optimal Arm', fontsize=12)
    plt.xticks(rotation=15, ha="right")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{int(yval)}', va='bottom', ha='center')
    plt.tight_layout()
    plt.show()


def run_s2():
    """
    Runs Experiment Set S2 and displays the resulting plot.
    """
    # --- S2 Configuration ---
    TIME_HORIZON = 30_000
    P_VALUES = np.round(np.arange(0.1, 0.901, 0.05), 2)
    print("\nS2: Running 17 games for 30,000 time steps each.")

    agents_to_test = {
        "Epsilon-Greedy (ε=0.1)": (EpsilonGreedyAgent, {"epsilon": 0.1}),
        "UCB1 (c=2)": (UCBAgent, {"c": 2}),
        "KL-UCB": (KLUCBAgent, {}),
        "Thompson Sampling": (ThompsonSamplingAgent, {}),
    }
    
    final_regrets = {name: [] for name in agents_to_test.keys()}

    # --- Main Simulation Loop for S2 ---
    for p in P_VALUES:
        bandit_probs = np.array([p, p + 0.1])
        print(f"  - Running game for p={p:.2f}, bandit_probs={bandit_probs}")
        for name, (agent_class, params) in agents_to_test.items():
            bandit = MultiArmedBandit(arms=bandit_probs)
            
            # Call the agent constructor using the 'time_horizon' keyword.
            agent = agent_class(time_horizon=TIME_HORIZON, bandit=bandit, **params)
            
            for _ in range(TIME_HORIZON):
                agent.give_pull()
            final_regret = bandit.cumulative_regret_array[-1]
            final_regrets[name].append(final_regret)

    print("S2: Simulations complete. Displaying plot...")

    # --- Plotting S2 Results ---
    plt.figure(figsize=(14, 8))
    for name, regrets in final_regrets.items():
        plt.plot(P_VALUES, regrets, marker='o', linestyle='-', label=name)

    plt.title('S2: Final Regret vs. Problem Difficulty (p)', fontsize=16)
    plt.xlabel('Value of p (Suboptimal Arm Probability)', fontsize=12)
    plt.ylabel(f'Final Regret after {TIME_HORIZON} Steps', fontsize=12)
    plt.xticks(P_VALUES, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("--- Starting Bandit Experiments ---")
    
    # Run Experiment Set 1
    run_s1()
    
    # Run Experiment Set 2
    run_s2()
    
    print("\nAll experiments complete.")

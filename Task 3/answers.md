# Analysis of S2 Results: The Regret Riddle

## The Observation

In the "S2: Final Regret vs. Problem Difficulty" plot, we observe a distinct and curious pattern for the UCB1 and KL-UCB algorithms. Instead of being flat or linear, their final regret forms an **inverted U-shape**, peaking when the value of `p` is around 0.4 or 0.45. This means the algorithms accumulate the most regret when the two arm probabilities are centered around 0.5 (e.g., `{0.4, 0.5}` or `{0.45, 0.55}`).

This is counter-intuitive at first. The difference in mean reward between the optimal and suboptimal arm is constant (`0.1`) across all 17 games. Why, then, is the problem "harder" for these agents in the middle range of `p`?

## The Explanation: The Role of Reward Variance

The answer lies in the **variance** of the rewards from the bandit arms.

For a bandit arm that pays a reward of 1 with probability `p` (a Bernoulli distribution), its variance is calculated as:

`Variance = p * (1 - p)`

Let's look at the shape of this variance function:
- If `p = 0.1`, Variance = 0.1 * 0.9 = 0.09
- If `p = 0.5`, Variance = 0.5 * 0.5 = 0.25 (This is the maximum possible variance)
- If `p = 0.9`, Variance = 0.9 * 0.1 = 0.09

The variance of the reward signal is highest when the probability `p` is close to 0.5.

### How Variance Affects UCB Algorithms

UCB1 and KL-UCB are "frequentist" algorithms that work by:
1.  Calculating the empirical (average) reward for each arm.
2.  Building a confidence interval (or an upper confidence bound) around this average.
3.  Choosing the arm with the highest upper bound.

The core challenge for these algorithms is to get a stable, accurate estimate of the true mean reward. **When the variance of the rewards is high, the observed average reward fluctuates more wildly from one pull to the next.** This statistical "noise" makes it much harder to distinguish the better arm from the worse one.

Consider two scenarios from the experiment:
1.  **Low Variance (p=0.1):** The arms are `{0.1, 0.2}`. The rewards are mostly 0s. When a '1' appears, it's a strong, informative signal. The empirical means stabilize relatively quickly, the confidence bounds shrink, and the agent quickly identifies the better arm.
2.  **High Variance (p=0.45):** The arms are `{0.45, 0.55}`. The rewards from both arms are a noisy mix of 0s and 1s. The empirical means jump around a lot. The confidence bounds for the two arms will overlap for a much longer time, causing the agent to make more "mistakes" by pulling the suboptimal arm while it tries to resolve the high uncertainty.

More mistakes (pulling the suboptimal arm) directly lead to higher cumulative regret. Therefore, the regret for UCB1 and KL-UCB is highest precisely where the underlying variance of the arms is highest—when their probabilities are close to 0.5.

### Comparison with Other Agents

-   **Thompson Sampling:** This Bayesian agent also shows a similar, though often less pronounced, hump. It is also affected by variance, as a higher variance leads to a wider posterior distribution, making the two distributions harder to distinguish. However, its probabilistic sampling nature can sometimes navigate this uncertainty more gracefully than the deterministic upper-bound selection of UCB.
-   **Epsilon-Greedy:** This agent's regret is dominated by its fixed exploration strategy. It explores by picking a random arm `epsilon` fraction of the time, regardless of uncertainty. Its regret is less sensitive to the underlying variance and more dependent on the fixed gap between the arms, which is why its final regret curve is much flatter.

In conclusion, the "difficulty" of a bandit problem for UCB-style algorithms is a function of both the **gap** between arm rewards and the **variance** of those rewards. In experiment S2, the gap was fixed, revealing the powerful and direct effect of variance on performance.

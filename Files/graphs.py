import numpy as np
import matplotlib.pyplot as plt
data = np.load('C:\\Users\\nicle\\Desktop\\GPBO-MBRL\\GP-MPC_NL\\PF_MPC_GP_Env\\ParallelOverParticles\\Jan24\\April 4th tests\\QRNN-MPC-main\\QRNN-MPC-main\\Files\\MuJoCoReacher_MPC_50NN_ASGNN_mid_results.npz')

episodic_rep_returns = data['episode_rewards']
mean_episodic_returns = data['mean_rewards']
std_episodic_returns = data['std_rewards']

plt.figure(1)
plt.plot(episodic_rep_returns, label='Episode Rewards')

plt.figure(2)
plt.plot(mean_episodic_returns, label='Mean Rewards')
plt.fill_between(range(len(mean_episodic_returns)), mean_episodic_returns - std_episodic_returns, mean_episodic_returns + std_episodic_returns, alpha=0.2, label='Std Dev')

plt.show()

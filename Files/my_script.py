# myScript.py
import numpy as np  # Test if scipy-stack loads
print("Hello, Compute Canada!")

rews = np.array([0., 0.])     # Shape: (reps,)

# Save to .npz file
np.savez(
    "MPC_QRNN_ASGNN_results.npz",
    episode_rewards=rews,
    )
    

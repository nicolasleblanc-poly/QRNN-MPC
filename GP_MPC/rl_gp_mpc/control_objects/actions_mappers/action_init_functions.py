import numpy as np


def generate_mpc_action_init_random(len_horizon, dim_action):
	return np.random.uniform(low=0, high=1, size=(len_horizon, dim_action)).reshape(-1)


def generate_mpc_action_init_frompreviousiter(actions_mpc, dim_action):
    # Shifts to remove the last action taken, the last action is untouche -> Duplicate of last action
	"""
	# Example
	actions = np.random.uniform(0, 2, size=(10))
	print(actions)
	action_dim = 2
	actions[:-action_dim] =  actions[action_dim:]
	print(actions)
	"""
	actions_mpc[:-dim_action] = actions_mpc[dim_action:]
	return actions_mpc


def get_init_action_change(len_horizon, max_change_action_norm):
	return np.dot(np.expand_dims(np.random.uniform(low=-1, high=1, size=(len_horizon)), 1), np.expand_dims(max_change_action_norm, 0))


def get_init_action(len_horizon, num_actions):
	return np.random.uniform(low=0, high=1, size=(len_horizon, num_actions))
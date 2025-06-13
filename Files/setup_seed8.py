import numpy as np
import torch
import gymnasium as gym
# import panda_gym

class setup_class:
    def __init__(self, prob, use_CEM=False):
        self.prob = prob

        self.nb_actions = None

        # Generate random seeds
        self.random_seeds = [8]
        # print("random_seeds ", type(random_seeds[0]), "\n")
        self.nb_rep_episodes = len(self.random_seeds)

        self.laplace_alpha = 1
        self.use_CEM = use_CEM

        # Constants
        self.batch_size = 32
        self.num_particles = 100
        self.quantiles = torch.tensor(np.array([0.0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]), dtype=torch.float32)
        self.num_quantiles = len(self.quantiles)
        self.nb_reps_MPC = 10
        self.nb_random = 0 # 10

        if prob == "CartPole":
            self.discrete = True
            self.horizon = 30
            # self.max_episodes = 100
            self.max_episodes = 400
            self.max_steps = 200

            # For test
            # self.max_episodes = 2
            # self.max_steps = 3
                        
            self.std = None
            # self.change_prob = 0.01
            # self.change_prob = 0.05
            self.change_prob = 0.1
            # self.change_prob = 0.3
            # self.change_prob = 0.5

            if self.change_prob == 0.01:
                self.change_prob_std = "001"
            elif self.change_prob == 0.05:
                self.change_prob_std = "005"
            elif self.change_prob == 0.1:
                self.change_prob_std = "01"
            elif self.change_prob == 0.3:
                self.change_prob_std = "03"
            elif self.change_prob == 0.5:
                self.change_prob_std = "05"    
      
            self.nb_actions = 2
            
            self.nb_top_particles = 5
            # nb_random = 10
            
            self.env = gym.make('CartPole-v0', render_mode="rgb_array").unwrapped # To save time since 200 instead of 500 steps per episode
            
            # Hyperparameters
            self.state_dim = self.env.observation_space.shape[0] # Since we only care about angle and omega which are given using env.state
            # action_dim = env.action_space.shape[0]  # For Pendulum, it's continuous
            self.action_dim = 1
            self.action_low = 0 #env.action_space.low[0]
            self.action_high = 1 # env.action_space.high[0]
            
            self.goal_state = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
            self.goal_state_dim = len(self.goal_state)

            self.states_low = torch.tensor([-4.8, -torch.inf, -0.41887903, -torch.inf])
            self.states_high = torch.tensor([4.8, torch.inf, 0.41887903, torch.inf])
            
            def compute_cost_CartPole(states, t, horizon, actions):
                """
                Vectorized cost computation for multiple states and actions.
                
                :param states: Tensor of shape (batch_size, state_dim)
                :param actions: Tensor of shape (batch_size,)
                :return: Cost tensor of shape (batch_size,)
                """
                cart_position = states[:, 0]
                pole_angle = states[:, 2]
                cart_velocity = states[:, 1]
                return pole_angle**2 + 0.1 * cart_position**2 + 0.1 * cart_velocity**2

            self.compute_cost_CartPole = compute_cost_CartPole

        elif prob == "Acrobot":
            self.discrete = True
            self.horizon = 30
            # self.max_episodes = 300
            self.max_episodes = 600
            self.max_steps = 200
            # self.max_steps = 600

            # For test
            # self.max_episodes = 2
            # self.max_steps = 3

            self.std = None
            # self.change_prob = 0.01
            # self.change_prob = 0.05
            # self.change_prob = 0.1
            self.change_prob = 0.3
            # self.change_prob = 0.5

            if self.change_prob == 0.01:
                self.change_prob_std = "001"
            elif self.change_prob == 0.05:
                self.change_prob_std = "005"
            elif self.change_prob == 0.1:
                self.change_prob_std = "01"
            elif self.change_prob == 0.3:
                self.change_prob_std = "03"
            elif self.change_prob == 0.5:
                self.change_prob_std = "05" 
            
            self.nb_actions = 3
            
            self.nb_top_particles = 5
            # nb_random = 10
            
            self.env = gym.make('Acrobot-v1', render_mode="rgb_array").unwrapped
            
            # Hyperparameters
            self.state_dim = self.env.observation_space.shape[0] # Since we only care about angle and omega which are given using env.state
            # action_dim = env.action_space.shape[0]  # For Pendulum, it's continuous
            self.action_dim = 1
            self.action_low = 0 #env.action_space.low[0]
            self.action_high = 2 # env.action_space.high[0]
            
            self.goal_state = torch.tensor([-1, 0, 1, 0, 0, 0], dtype=torch.float32)
            self.goal_state_dim = len(self.goal_state)

            self.states_low = torch.tensor([-1, -1, -1, -1, -12.566371, -28.274334])
            self.states_high = torch.tensor([1, 1, 1, 1, 12.566371, 28.274334])
            
            def compute_cost_Acrobot(states, t, horizon, actions):
                """
                Compute the cost based on the state and action for the CartPole environment.
                """

                # theta1 = np.arccos(np.clip(states[:, 0],-1,1))  # First joint angle
                # theta2 = np.arccos(np.clip(states[:, 2],-1,1))  # Second joint angle
                
                # distance_to_goal = 1 + (np.cos(theta1) + np.cos(theta1 + theta2))
                
                theta1 = torch.arccos(torch.clip(states[:, 0],-1,1))  # First joint angle
                theta2 = torch.arccos(torch.clip(states[:, 2],-1,1))  # Second joint angle

                
                # states_low = torch.tensor([-1, -1, -1, -1, -12.566371, -28.274334])
                # states_high = torch.tensor([1, 1, 1, 1, 12.566371, 28.274334])
                # theta1 = 2 * ((states[:, 0] - states_low) / (states_high - states_low)) - 1
                # theta2 = 2 * ((states[:, 2] - states_low) / (states_high - states_low)) - 1
                
                
                distance_to_goal = 1 + (torch.cos(theta1) + torch.cos(theta1 + theta2))
                cost = distance_to_goal ** 2
                
                return cost
            
            self.compute_cost_Acrobot = compute_cost_Acrobot
            
        elif prob == "LunarLander": # ToDo
            self.discrete = True
            self.horizon = 30
            self.max_episodes = 400
            # max_steps = 200 # No defined max episode length
            self.max_steps = 1000
            self.std = None
            # self.change_prob = 0.01
            # self.change_prob = 0.05
            # self.change_prob = 0.1
            self.change_prob = 0.3
            # self.change_prob = 0.5

            if self.change_prob == 0.01:
                self.std_string = "001"
            elif self.change_prob == 0.05:
                self.std_string = "005"
            elif self.change_prob == 0.1:
                self.std_string = "01"
            elif self.change_prob == 0.3:
                self.std_string = "03"
            elif self.change_prob == 0.5:
                self.std_string = "05"
                

            self.nb_actions = 3
            
            self.nb_top_particles = 5
            # nb_random = 10
            
            self.env = gym.make('LunarLander-v3', continuous=False, render_mode="rgb_array").unwrapped
            
            # Hyperparameters
            self.state_dim = self.env.observation_space.shape[0] # Since we only care about angle and omega which are given using env.state
            # action_dim = env.action_space.shape[0]  # For Pendulum, it's continuous
            self.action_dim = 1
            self.action_low = 0 #env.action_space.low[0]
            self.action_high = 2 # env.action_space.high[0]
            
            self.goal_state = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.float32)
            # self.goal_state = torch.tensor([0, -0.01, 0, 0, 0, 0, 1, 1], dtype=torch.float32)
            self.goal_state_dim = len(self.goal_state)

            self.states_low = torch.tensor([-2.5, -2.5, -10, -10, -6.2831855, -10, 0, 0])
            self.states_high = torch.tensor([2.5, 2.5, 10, 10, 6.2831855, 10, 1, 1])
            
            def compute_cost_LunarLander(states, t, horizon, actions):
                """
                Compute the cost based on the state and action for the CartPole environment.
                """


                x = states[:, 0]
                y = states[:, 1]
                vx = states[:, 2]
                vy = states[:, 3]
                theta = states[:, 4]
                omega = states[:, 5]
                leg1 = states[:, 6]
                leg2 = states[:, 7]
                # legs = states[:, 6:8]

                m_power = (actions[:, 0] == 2).float()
                s_power = ((actions[:, 0] == 1) | (actions[:, 0] == 3)).float()
                
                position_cost = 100*np.sqrt(x**2 + y**2)  # Penalize distance from the origin
                velocity_cost = 0.1 * (vx**2 + vy**2)  # Penalize high velocities
                angle_cost = 10 * theta**2  # Penalize deviation from upright position
                contact_bonus = -10 * (leg1 + leg2)  # Reward for having legs in contact with the ground
                fuel_cost = 0.3*m_power + 0.03*s_power  # Penalize fuel consumption
                return position_cost + velocity_cost + angle_cost + contact_bonus + fuel_cost

                # Part of old cost function
                # legs = states[:, 6:8]
                # a1, a2 = actions[:, 0], actions[:, 1]
                
                # cost = 10*x**2 + 10*y**2 + vx**2 + vy**2 + theta**2 + omega**2 - 10*legs.sum(dim=1) #+ 0.1 * (a1 ** 2 + a2 ** 2)
                # cost = distance_to_goal ** 2
                
                return cost
            
            self.compute_cost_LunarLander = compute_cost_LunarLander

        elif prob == "MountainCar": # ToDo
            self.discrete = True
            # self.horizon = 70 # Value before change to 30
            self.horizon = 30
            # horizon = 100
            # max_episodes = 100
            
            # self.max_episodes = 10
            self.max_episodes = 500
            self.max_steps = 200
            self.std = None
            # change_prob = 0.01
            # change_prob = 0.05
            self.change_prob = 0.1
            # self.change_prob = 0.3
            # self.change_prob = 0.5

            if self.change_prob == 0.01:
                self.change_prob_std = "001"
            elif self.change_prob == 0.05:
                self.change_prob_std = "005"
            elif self.change_prob == 0.1:
                self.change_prob_std = "01"
            elif self.change_prob == 0.3:
                self.change_prob_std = "03"
            elif self.change_prob == 0.5:
                self.change_prob_std = "05" 
            
            self.nb_top_particles = 5
            # nb_random = 10

            self.nb_actions = 3
            
            self.env = gym.make('MountainCar-v0', render_mode="rgb_array").unwrapped
            
            # Hyperparameters
            self.state_dim = self.env.observation_space.shape[0] # Since we only care about angle and omega which are given using env.state
            # action_dim = env.action_space.shape[0]  # For Pendulum, it's continuous
            self.action_dim = 1
            self.action_low = 0
            self.action_high = 3
            
            # goal_state = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
            self.goal_state = torch.tensor([0.5], dtype=torch.float32)
            self.goal_state_dim = len(self.goal_state)

            self.states_low = torch.tensor([-1.2, -0.07])
            self.states_high = torch.tensor([0.6, 0.07])
            
            def compute_cost_MountainCar(states, t, horizon, actions):
                actions = actions.reshape(-1)

                goal_position = 0.5  # Position goal in Mountain Car
                # gamma = 0.90 # 0.0  # Discount factor for delayed rewards
                gamma = 0.5 # Try this value
                # Or only compare the last position
                
                # Distance reward: Encourage progress towards the goal
                distance_reward = (states[:, 0]-goal_position)**2 # -abs(goal_position - state[0])

                reverse_discount_factor = gamma**(horizon-t-1)
                distance_reward = reverse_discount_factor*distance_reward # -0.1*states[:, 1]**2
                # distance_reward += 0.05*(actions)**2

                # weights = np.array([1, 0])
                # distance_reward = np.dot(weights, distance_reward)

                return distance_reward
                
                # return reverse_discount_factor*distance_reward
            
            self.compute_cost_MountainCar = compute_cost_MountainCar

        elif prob == "InvertedPendulum":
            self.discrete = False
            self.horizon = 30
            self.max_episodes = 300
            self.max_steps = 1000

            # For test
            # self.max_episodes = 2
            # self.max_steps = 3

            # Current test values
            # self.std = 0
            self.std = 3e-1
            # self.std = 1.5

            # Older test values
            # std = 1e-1
            # std = 3e-1
            # std = 1
            # std = 1.5
            self.change_prob = None

            # self.std_string = "0"
            self.std_string = "3em1"
            # std_string = "15"
            
            self.nb_top_particles = 5
            # nb_random = 10
            
            self.env = gym.make('InvertedPendulum-v5', render_mode="rgb_array").unwrapped
            
            # Hyperparameters
            self.state_dim = self.env.observation_space.shape[0]
            # state_dim = env.observation_space.shape[0]-1 # Since we only care about angle and omega which are given using env.state
            # action_dim = self.env.action_space.shape[0]  # For Pendulum, it's continuous
            self.action_dim = 1
            self.action_low = self.env.action_space.low[0]
            self.action_high = self.env.action_space.high[0]
            
            self.goal_state = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
            self.goal_state_dim = len(self.goal_state)

            self.states_low = torch.tensor([-100, -1, -100, -100])
            self.states_high = torch.tensor([100, 1, 100, 100])
            
            def compute_cost_InvertedPendulum(states, t, horizon, actions):
                """
                Vectorized cost computation for multiple states and actions.
                
                :param states: Tensor of shape (batch_size, state_dim)
                :param actions: Tensor of shape (batch_size,)
                :return: Cost tensor of shape (batch_size,)
                """
                cart_position = states[:, 0]
                pole_angle = states[:, 1] # Opposite of cart pole
                cart_velocity = states[:, 2] # Opposite of cart pole
                return pole_angle**2 + 0.1 * cart_position**2 + 0.1 * cart_velocity**2

            self.compute_cost_InvertedPendulum = compute_cost_InvertedPendulum

        elif prob == "Pendulum_xyomega":
            self.discrete = False
            self.horizon = 15
            self.max_episodes = 300
            self.max_steps = 200

            # For test
            # self.max_episodes = 2
            # self.max_steps = 3

            # Current test values
            # self.std = 0
            self.std = 3e-1
            # self.std = 1.5

            # Older test values
            # std = 1e-1
            # std = 3e-1
            # std = 1
            # std = 1.5
            self.change_prob = None

            # self.std_string = "0"
            self.std_string = "3em1"
            # std_string = "15"
            
            self.nb_top_particles = 5
            # nb_random = 10
            
            self.env = gym.make('Pendulum-v1', render_mode="rgb_array").unwrapped
            
            # Hyperparameters
            self.state_dim = self.env.observation_space.shape[0]
            # state_dim = env.observation_space.shape[0]-1 # Since we only care about angle and omega which are given using env.state
            # action_dim = env.action_space.shape[0]  # For Pendulum, it's continuous
            self.action_dim = 1
            self.action_low = self.env.action_space.low[0]
            self.action_high = self.env.action_space.high[0]
            
            self.goal_state = torch.tensor([0, 0], dtype=torch.float32)
            self.goal_state_dim = len(self.goal_state)

            self.states_low = torch.tensor([-1, -1, -8])
            self.states_high = torch.tensor([1, 1, 8])
            
            def angle_normalize(x):
                return ((x + np.pi) % (2 * np.pi)) - np.pi
            
            def compute_cost_Pendulum_xy_omega(states, t, horizon, actions):
                """
                Vectorized cost function for all particles.

                :param states: Tensor of shape [num_particles, state_dim] containing (theta, omega).
                :param t: Current time step (scalar).
                :param horizon: Planning horizon (scalar).
                :param actions: Tensor of shape [num_particles, 1] (or [num_particles,]) containing actions.
                :return: Tensor of shape [num_particles] with discounted costs.
                """
                gamma = 0.99  # Discount factor

                # Extract theta and omega
                # theta = states[:, 0]#.detach().numpy()  # Shape: [num_particles]
                # omega = states[:, 1]# .detach().numpy()  # Shape: [num_particles]

                x = states[:, 0]
                y = states[:, 1]
                omega = states[:, 2]
                
                # print("actions.shape ", actions.detach().numpy().reshape(len(actions)).shape, "\n")
                # print("theta.shape ", theta.shape, "\n")
                # print("omega.shape ", omega.shape, "\n")
                # print("actions ", actions, "\n")

                # Normalize theta to [-π, π]
                # theta = (theta + torch.pi) % (2 * torch.pi) - torch.pi 

                # Ensure actions have the correct shape
                # actions = actions.view(-1)  # Flatten to shape [num_particles]

                # Compute cost function
                # cost = theta ** 2 + 0.1 * omega ** 2 + 0.01 * actions.reshape(len(actions)) ** 2
                cost = (1-x)**2 + y**2 + 0.1 * omega**2 + 0.01 * actions.reshape(len(actions)) ** 2

                # Compute discount factor
                reverse_discount_factor = gamma ** (horizon - t - 1)
                
                # print("cost ", cost, "\n")

                return cost * reverse_discount_factor  # Shape: [num_particles]
            
            self.compute_cost_Pendulum_xy_omega = compute_cost_Pendulum_xy_omega


        elif prob == "CartPoleContinuous":
            self.discrete = False
            self.horizon = 30
            # self.max_episodes = 100
            self.max_episodes = 400
            self.max_steps = 200

            # For test
            # self.max_episodes = 2
            # self.max_steps = 3
                        
            # Current test values
            # self.std = 0
            self.std = 3e-1
            # self.std = 1.5
            
            self.change_prob = None

            # if self.change_prob == 0.01:
            #     self.change_prob_std = "001"
            # elif self.change_prob == 0.05:
            #     self.change_prob_std = "005"
            # elif self.change_prob == 0.1:
            #     self.change_prob_std = "01"
            # elif self.change_prob == 0.3:
            #     self.change_prob_std = "03"
            # elif self.change_prob == 0.5:
            #     self.change_prob_std = "05"    
      
            # self.nb_actions = 2
            
            self.nb_top_particles = 5
            # nb_random = 10
            
            import cartpole_continuous as cartpole_env
            
            self.env = cartpole_env.CartPoleContinuousEnv(render_mode="rgb_array").unwrapped # To save time since 200 instead of 500 steps per episode
            
            # self.env = gym.make('CartPole-v0', render_mode="rgb_array").unwrapped # To save time since 200 instead of 500 steps per episode
            
            # Hyperparameters
            self.state_dim = self.env.observation_space.shape[0] # Since we only care about angle and omega which are given using env.state
            # action_dim = env.action_space.shape[0]  # For Pendulum, it's continuous
            self.action_dim = 1
            self.action_low = self.env.action_space.low[0]
            self.action_high = self.env.action_space.high[0]
            
            self.goal_state = torch.tensor([0, 0], dtype=torch.float32)
            self.goal_state_dim = len(self.goal_state)

            self.states_low = torch.tensor([-4.8, -torch.inf, -0.41887903, -torch.inf])
            self.states_high = torch.tensor([4.8, torch.inf, 0.41887903, torch.inf])
            
            def compute_cost_CartPoleContinuous(states, t, horizon, actions):
                """
                Vectorized cost computation for multiple states and actions.
                
                :param states: Tensor of shape (batch_size, state_dim)
                :param actions: Tensor of shape (batch_size,)
                :return: Cost tensor of shape (batch_size,)
                """
                cart_position = states[:, 0]
                pole_angle = states[:, 2]
                cart_velocity = states[:, 1]
                return pole_angle**2 + 0.1 * cart_position**2 + 0.1 * cart_velocity**2

            self.compute_cost_CartPoleContinuous = compute_cost_CartPoleContinuous

        elif prob == "Pendulum":
            self.discrete = False
            self.horizon = 15
            self.max_episodes = 300
            self.max_steps = 200

            # For test
            # self.max_episodes = 3
            # self.max_steps = 5

            # Current test values
            self.std = 0
            # self.std = 3e-1
            # self.std = 1.5

            # Older test values
            # std = 1e-1
            # std = 3e-1
            # std = 1
            # std = 1.5
            self.change_prob = None

            self.std_string = "0"
            
            self.nb_top_particles = 5
            # nb_random = 10
            
            self.env = gym.make('Pendulum-v1', render_mode="rgb_array").unwrapped
            
            # Hyperparameters
            self.state_dim = self.env.observation_space.shape[0]-1 # Since we only care about angle and omega which are given using env.state
            # action_dim = env.action_space.shape[0]  # For Pendulum, it's continuous
            self.action_dim = 1
            self.action_low = self.env.action_space.low[0]
            self.action_high = self.env.action_space.high[0]
            
            self.goal_state = torch.tensor([0, 0], dtype=torch.float32)
            self.goal_state_dim = len(self.goal_state)

            self.states_low = torch.tensor([-1, -1, -8])
            self.states_high = torch.tensor([1, 1, 8])
            
            def angle_normalize(x):
                return ((x + np.pi) % (2 * np.pi)) - np.pi
            
            def compute_cost_Pendulum(states, t, horizon, actions):
                """
                Vectorized cost function for all particles.

                :param states: Tensor of shape [num_particles, state_dim] containing (theta, omega).
                :param t: Current time step (scalar).
                :param horizon: Planning horizon (scalar).
                :param actions: Tensor of shape [num_particles, 1] (or [num_particles,]) containing actions.
                :return: Tensor of shape [num_particles] with discounted costs.
                """
                gamma = 0.99  # Discount factor

                # Extract theta and omega
                theta = states[:, 0]#.detach().numpy()  # Shape: [num_particles]
                omega = states[:, 1]# .detach().numpy()  # Shape: [num_particles]
                
                # print("actions.shape ", actions.detach().numpy().reshape(len(actions)).shape, "\n")
                # print("theta.shape ", theta.shape, "\n")
                # print("omega.shape ", omega.shape, "\n")
                # print("actions ", actions, "\n")

                # Normalize theta to [-π, π]
                theta = (theta + torch.pi) % (2 * torch.pi) - torch.pi  

                # Ensure actions have the correct shape
                # actions = actions.view(-1)  # Flatten to shape [num_particles]

                # Compute cost function
                cost = theta ** 2 + 0.1 * omega ** 2 + 0.01 * actions.reshape(len(actions)) ** 2

                # Compute discount factor
                reverse_discount_factor = gamma ** (horizon - t - 1)
                
                # print("cost ", cost, "\n")

                return cost * reverse_discount_factor  # Shape: [num_particles]
            
            self.compute_cost_Pendulum = compute_cost_Pendulum
            
        elif prob == "MountainCarContinuous":
            self.discrete = False
            # self.horizon = 70
            # self.horizon = 12 # Value before change to 30
            self.horizon = 30
            # horizon = 100
            # self.max_episodes = 100
            self.max_episodes = 500
            # self.max_steps = 300 # 999 # in reality
            self.max_steps = 1000
            # std = 1e-2
            # self.std = 1e-1
            self.std = 3e-1
            # self.std = 6e-1
            self.change_prob = None
            
            self.nb_top_particles = 5
            # nb_random = 10
            
            self.env = gym.make('MountainCarContinuous-v0', render_mode="rgb_array").unwrapped
            
            # Hyperparameters
            self.state_dim = self.env.observation_space.shape[0] # Since we only care about angle and omega which are given using env.state
            # action_dim = env.action_space.shape[0]  # For Pendulum, it's continuous
            self.action_dim = 1
            self.action_low = self.env.action_space.low[0]
            self.action_high = self.env.action_space.high[0]
            
            # goal_state = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
            self.goal_state = torch.tensor([0.45], dtype=torch.float32)
            self.goal_state_dim = len(self.goal_state)

            self.states_low = torch.tensor([-1.2, -0.07])
            self.states_high = torch.tensor([0.6, 0.07])
            
            def compute_cost_MountainCarContinuous(states, t, horizon, actions):
                
                # print("actions.shape ", actions.shape, "\n")
                actions = actions.reshape(-1)

                goal_position = 0.45  # Position goal in Mountain Car
                # gamma = 0.90 # 0.0  # Discount factor for delayed rewards
                gamma = 0.5 # Try this value
                # Or only compare the last position
                
                # Distance reward: Encourage progress towards the goal
                distance_reward = (states[:, 0]-goal_position)**2 # -abs(goal_position - state[0])
                
                reverse_discount_factor = gamma**(horizon-t-1)
                distance_reward = reverse_discount_factor*distance_reward #-0.1*states[:, 1]**2
                # distance_reward += 0.05*(actions)**2
                return distance_reward
                # return reverse_discount_factor*distance_reward
            
            self.compute_cost_MountainCarContinuous = compute_cost_MountainCarContinuous

        elif prob == "LunarLanderContinuous": # ToDo
            self.discrete = False
            self.horizon = 30
            self.max_episodes = 300
            # self.max_steps = 200 # No defined max episode length
            self.max_steps = 1000

            # For test
            # self.max_episodes = 2
            # self.max_steps = 3

            # self.std = 1e-1
            self.std = 3e-1
            # self.std = 1
            # self.std = 1.5
            self.change_prob = None
            
            self.nb_top_particles = 5
            # nb_random = 10
            
            self.env = gym.make('LunarLander-v3', continuous=True, render_mode="rgb_array").unwrapped
            
            # Hyperparameters
            self.state_dim = self.env.observation_space.shape[0] # Since we only care about angle and omega which are given using env.state
            # action_dim = env.action_space.shape[0]  # For Pendulum, it's continuous
            self.action_dim = 2
            self.action_low = -1 #env.action_space.low[0]
            self.action_high = 1 # env.action_space.high[0]
            
            self.goal_state = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.float32)
            # self.goal_state = torch.tensor([0, -0.01, 0, 0, 0, 0, 1, 1], dtype=torch.float32)
            self.goal_state_dim = len(self.goal_state)

            self.states_low = torch.tensor([-2.5, -2.5, -10, -10, -6.2831855, -10, 0, 0])
            self.states_high = torch.tensor([2.5, 2.5, 10, 10, 6.2831855, 10, 1, 1])
            
            def compute_cost_LunarLanderContinuous(states, t, horizon, actions):
                """
                Compute the cost based on the state and action for the CartPole environment.
                """
                
                x = states[:, 0]
                y = states[:, 1]
                vx = states[:, 2]
                vy = states[:, 3]
                theta = states[:, 4]
                omega = states[:, 5]
                leg1 = states[:, 6]
                leg2 = states[:, 7]
                # legs = states[:, 6:8]

                main_thrust = actions[:, 0]  # Main engine thrust
                side_thrust = actions[:, 1]
                # a1, a2 = actions[:, 0], actions[:, 1]
                # m_power calculation
                clipped_main = torch.clamp(main_thrust, 0.0, 1.0)
                m_power = torch.where(
                    main_thrust > 0.0,
                    (clipped_main + 1.0) * 0.5,
                    torch.zeros_like(main_thrust)
                )

                # s_power calculation
                abs_side = torch.abs(side_thrust)
                clipped_side = torch.clamp(abs_side, 0.5, 1.0)
                s_power = torch.where(
                    abs_side > 0.5,
                    clipped_side,
                    torch.zeros_like(side_thrust)
                )
                
                position_cost = 100*np.sqrt(x**2 + y**2)  # Penalize distance from the origin
                velocity_cost = 0.1 * (vx**2 + vy**2)  # Penalize high velocities
                angle_cost = 10 * theta**2  # Penalize deviation from upright position
                contact_bonus = -10 * (leg1 + leg2)  # Reward for having legs in contact with the ground
                fuel_cost = 0.3*m_power + 0.03*s_power  # Penalize fuel consumption
                return position_cost + velocity_cost + angle_cost + contact_bonus + fuel_cost

                # Part of old cost function
                # legs = states[:, 6:8]
                # cost = 10*x**2 + 10*y**2 + vx**2 + vy**2 + theta**2 + omega**2 - 10*legs.sum(dim=1) + 0.1 * (a1 ** 2 + a2 ** 2)
                # # cost = distance_to_goal ** 2
                
                return cost
                
                # x = states[:, 0]
                # y = states[:, 1]
                # vx = states[:, 2]
                # vy = states[:, 3]
                # theta = states[:, 4]
                # omega = states[:, 5]
                # # leftcontact = states[:, 6]
                # # rightcontact = states[:, 7]
                # legs = states[:, 6:8]
                # a1, a2 = actions[:, 0], actions[:, 1]
                
                # cost = 10*x**2 + 10*y**2 + vx**2 + vy**2 + theta**2 + omega**2 - 10*legs.sum(dim=1) + 0.1 * (a1 ** 2 + a2 ** 2)
                # # cost = distance_to_goal ** 2
                
                # return cost
            
            self.compute_cost_LunarLanderContinuous = compute_cost_LunarLanderContinuous

        elif prob == "PandaReacher":
            self.discrete = False
            self.horizon = 15
            # self.max_episodes = 100
            self.max_episodes = 400
            self.max_steps = 50
            
            # For test
            # self.max_episodes = 3
            # self.max_steps = 5
            
            # self.std = 1e-1
            self.std = 3e-1
            # self.std = 1
            # self.std = 1.5
            self.change_prob = None

            if self.std == 1e-1:
                self.std_string = "1em1"
            elif self.std == 3e-1:
                self.std_string = "3em1"
            elif self.std == 1:
                self.std_string = "1"
            elif self.std == 1.5:
                self.std_string = "15"
            
            self.goal_state = None # Defined when resetting the env
            
            self.nb_top_particles = 5
            # nb_random = 10
            
            self.env = gym.make('PandaReach-v3', render_mode="rgb_array").unwrapped # Reward only when the end effector is at the goal position
            # env = gym.make('PandaReachDense-v3', render_mode='human').unwrapped # Reward at each time step based on the distance to the goal position
            
            # Hyperparameters    
            self.actions_low = self.env.action_space.low#[0] #[:3]
            self.actions_high = self.env.action_space.high#[0] #[:3]
            self.states_low = self.env.observation_space['observation'].low#[:3]
            self.states_high = self.env.observation_space['observation'].high#[:3]
            self.state_dim = len(self.states_low)
            self.action_dim = len(self.actions_low)
            self.action_low = self.actions_low[0]
            self.action_high = self.actions_high[0]
            self.goal_state_dim = 3 #len(states_low)
            
            self.states_low = torch.tensor([-10, -10, -10, -10, -10, -10])
            self.states_high = torch.tensor([10, 10, 10, 10, 10, 10])
            
            def compute_cost_PandaReacher(states, t, horizon, actions, goal_state=None):
                # print("states ", states, "\n")
                # print("states[:, :3] ", states[:, :3], "\n")
                goal_state = torch.tensor(goal_state, dtype=torch.float32, device=states.device).reshape(1, 3)
                costs = torch.norm(states[:, :3] - goal_state, dim=1)

                # print("goal_state ", goal_state, "\n")
                # print("costs ", costs, "\n")
                # print("cost 0 ", torch.norm(states[0, :3]-torch.tensor(goal_state, dtype=torch.float32)), "\n")
                # print("cost 1 ", torch.norm(states[1, :3]-torch.tensor(goal_state, dtype=torch.float32)), "\n")
                return costs
                # torch.norm(states[:, :3]-torch.tensor(goal_state, dtype=torch.float32), dim=1)# +0.1*(torch.norm(actions))**2

            self.compute_cost_PandaReacher = compute_cost_PandaReacher

        elif prob == "PandaReacherDense":
            self.discrete = False
            self.horizon = 15
            # self.max_episodes = 100
            self.max_episodes = 400
            self.max_steps = 50
            
            # For test
            # self.max_episodes = 3
            # self.max_steps = 5
            
            # self.std = 1e-1
            self.std = 3e-1
            # self.std = 1
            # self.std = 1.5
            self.change_prob = None

            if self.std == 1e-1:
                self.std_string = "1em1"
            elif self.std == 3e-1:
                self.std_string = "3em1"
            elif self.std == 1:
                self.std_string = "1"
            elif self.std == 1.5:
                self.std_string = "15"
            
            self.goal_state = None # Defined when resetting the env
            
            self.nb_top_particles = 5
            # nb_random = 10
            
            self.env = gym.make('PandaReachDense-v3', render_mode="rgb_array").unwrapped # Reward only when the end effector is at the goal position
            # env = gym.make('PandaReachDense-v3', render_mode='human').unwrapped # Reward at each time step based on the distance to the goal position
            
            # Hyperparameters    
            self.actions_low = self.env.action_space.low#[0] #[:3]
            self.actions_high = self.env.action_space.high#[0] #[:3]
            self.states_low = self.env.observation_space['observation'].low#[:3]
            self.states_high = self.env.observation_space['observation'].high#[:3]
            self.state_dim = len(self.states_low)
            self.action_dim = len(self.actions_low)
            self.action_low = self.actions_low[0]
            self.action_high = self.actions_high[0]
            self.goal_state_dim = 3 #len(states_low)
            
            self.states_low = torch.tensor([-10, -10, -10, -10, -10, -10])
            self.states_high = torch.tensor([10, 10, 10, 10, 10, 10])
            
            def compute_cost_PandaReacherDense(states, t, horizon, actions, goal_state=None):
                # print("states ", states, "\n")
                # print("states[:, :3] ", states[:, :3], "\n")
                goal_state = torch.tensor(goal_state, dtype=torch.float32, device=states.device).reshape(1, 3)
                costs = torch.norm(states[:, :3] - goal_state, dim=1)

                # print("goal_state ", goal_state, "\n")
                # print("costs ", costs, "\n")
                # print("cost 0 ", torch.norm(states[0, :3]-torch.tensor(goal_state, dtype=torch.float32)), "\n")
                # print("cost 1 ", torch.norm(states[1, :3]-torch.tensor(goal_state, dtype=torch.float32)), "\n")
                return costs
                # torch.norm(states[:, :3]-torch.tensor(goal_state, dtype=torch.float32), dim=1)# +0.1*(torch.norm(actions))**2

            self.compute_cost_PandaReacherDense = compute_cost_PandaReacherDense

        elif prob == "PandaPusher": # ToDo
            self.discrete = False
            self.horizon = 15
            self.max_episodes = 400
            self.max_steps = 50

            # For test
            # self.max_episodes = 3
            # self.max_steps = 5

            # self.std = 1e-1
            self.std = 3e-1
            # self.std = 1
            # self.std = 1.5
            self.change_prob = None
            
            self.std_string = "1em1"
            
            self.goal_state = None # Defined when resetting the env
            
            self.nb_top_particles = 5
            # nb_random = 10
            
            # env = gym.make("PandaPush-v3").unwrapped # Reward only when the end effector is at the goal position
            self.env = gym.make('PandaPushDense-v3', render_mode="rgb_array").unwrapped # Reward at each time step based on the distance to the goal position
            
            # Hyperparameters    
            self.actions_low = self.env.action_space.low #[:3]
            self.actions_high = self.env.action_space.high #[:3]
            self.states_low = self.env.observation_space['observation'].low #[:3]
            self.states_high = self.env.observation_space['observation'].high #[:3]
            self.state_dim = len(self.states_low)
            self.action_dim = len(self.actions_low)
            self.action_low = self.actions_low[0]
            self.action_high = self.actions_high[0]
            self.goal_state_dim = 3 #len(states_low)

            self.states_low = torch.tensor([-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10])
            self.states_high = torch.tensor([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
            
            def compute_cost_PandaPusher(states, t, horizon, actions, goal_state=None):
                # print("goal_state ", goal_state, "\n")
                # print("states[0] ", states[0], "\n")
                # print("states.shape ", states.shape, "\n")
                # print("states[:, :3].shape ", states[:, :3].shape, "\n")
                # print("states[:, 6:9].shape ", states[:, 6:9].shape, "\n")
                
                goal_state = torch.tensor(goal_state, dtype=torch.float32, device=states.device).reshape(1, 3)
                end_effector_vel = states[:, 3:6]
                costs = torch.norm(states[:, :3]-states[:, 6:9], dim=1)+torch.norm(states[:, 6:9]-goal_state, dim=1)+0.01*(torch.norm(end_effector_vel, dim=1))**2
                
                # print("cost1 ", torch.norm(states[0, :3]-states[0, 6:9])+torch.norm(states[0, 6:9]-goal_state), '\n')
                # print("cost2 ", torch.norm(states[1, :3]-states[1, 6:9])+torch.norm(states[1, 6:9]-goal_state), '\n')
                # print("costs ", costs, "\n")
                
                return costs
            
            self.compute_cost_PandaPusher = compute_cost_PandaPusher

        elif prob == "MuJoCoReacher":
            self.discrete = False
            self.horizon = 15
            self.max_episodes = 400
            self.max_steps = 50 

            # For test
            # self.max_episodes = 3
            # self.max_steps = 5

            # self.std = 1e-1
            self.std = 3e-1
            # self.std = 1
            # self.std = 1.5
            self.change_prob = None

            if self.std == 1e-1:
                self.std_string = "1em1"
            elif self.std == 3e-1:
                self.std_string = "3em1"
            elif self.std == 1:
                self.std_string = "1"
            elif self.std == 1.5:
                self.std_string = "15"
            
            self.goal_state = None # Defined when resetting the env
            
            self.nb_top_particles = 5
            # nb_random = 10
            
            self.env = gym.make('Reacher-v5', render_mode="rgb_array").unwrapped
            
            # Hyperparameters    
            self.actions_low = self.env.action_space.low#[:3]
            self.actions_high = self.env.action_space.high#[:3]
            # states_low = env.observation_space['observation'].low#[:3]
            # states_high = env.observation_space['observation'].high#[:3]
            self.state_dim = 8
            self.action_dim = len(self.actions_low)
            
            self.action_low = self.actions_low[0]
            self.action_high = self.actions_high[0]
            
            self.goal_state_dim = 2

            self.states_low = torch.tensor([-1, -1, -1, -1, -100, -100, -torch.inf, -torch.inf])
            self.states_high = torch.tensor([1, 1, 1, 1, 100, 100, torch.inf, torch.inf])
            
            def compute_cost_MuJoCoReacher(states, t, horizon, actions, goal_state=None):
                # cost1 = torch.sqrt(states[0, -2]**2+states[0, -1]**2)+0.1*(torch.norm(actions[0]))**2
                # costs2 = torch.sqrt(states[1, -2]**2+states[1, -1]**2)+0.1*(torch.norm(actions[1]))**2
                costs = torch.sqrt(states[:, -2]**2+states[:, -1]**2)+0.1*(torch.norm(actions, dim=1))**2
                # print("cost1 ", cost1, "\n")
                # print("costs2 ", costs2, "\n")
                # print("costs ", costs, "\n")
                return costs
            # torch.norm(states[:, :3]-torch.tensor(goal_state, dtype=torch.float32))

            self.compute_cost_MuJoCoReacher = compute_cost_MuJoCoReacher

        elif prob == "MuJoCoPusher": # ToDo 
            self.discrete = False
            self.horizon = 15
            self.max_episodes = 400
            self.max_steps = 50 

            # For test
            # self.max_episodes = 3
            # self.max_steps = 5

            # self.std = 1e-1
            self.std = 3e-1
            # self.std = 1
            # self.std = 1.5
            self.change_prob = None
            
            self.goal_state = None # Defined when resetting the env
            
            self.nb_top_particles = 5
            # nb_random = 10
            
            self.env = gym.make('Pusher-v5', render_mode="rgb_array").unwrapped

            self.states_low = torch.tensor([-100]*23)
            self.states_high = torch.tensor([100]*23)
            
            # Hyperparameters    
            self.actions_lows = self.env.action_space.low#[:3]
            self.actions_highs = self.env.action_space.high#[:3]
            # self.states_low = self.env.observation_space['observation'].low#[:3]
            # self.states_high = self.env.observation_space['observation'].high#[:3]
            # self.state_dim = len(self.states_low)
            self.state_dim = 23
            self.action_dim = len(self.actions_lows)
            
            self.action_low = self.actions_lows[0]
            self.action_high = self.actions_highs[0]
            
            self.goal_state_dim = 3

            # TBD
            # self.states_low = None # torch.tensor([-1, -1, -1, -1, -100, -100, -torch.inf, -torch.inf])
            # self.states_high = None # torch.tensor([1, 1, 1, 1, 100, 100, torch.inf, torch.inf])
            
            def compute_cost_MuJoCoPusher(states, t, horizon, actions, goal_state=None):
                return torch.norm(states[:, 14:17]-states[:, 17:20], dim=1)+torch.norm(states[:, 17:20]-states[:, 20:], dim=1)
                
                # return torch.sqrt(states[:, -2]**2+states[:, -1]**2)+0.1*(torch.norm(actions))**2
            
            # torch.norm(states[:, :3]-torch.tensor(goal_state, dtype=torch.float32))

            self.compute_cost_MuJoCoPusher = compute_cost_MuJoCoPusher

    def compute_cost(self, prob, states, t, horizon, actions, goal_state=None):
        if prob == "CartPole":
            return self.compute_cost_CartPole(states, t, horizon, actions)
        elif prob == "Acrobot":
            return self.compute_cost_Acrobot(states, t, horizon, actions)
        elif prob == "LunarLander":
            return self.compute_cost_LunarLander(states, t, horizon, actions)
        elif prob == "MountainCar":
            return self.compute_cost_MountainCar(states, t, horizon, actions)
        elif prob == "InvertedPendulum":
            return self.compute_cost_InvertedPendulum(states, t, horizon, actions)
        elif prob == "Pendulum":
            return self.compute_cost_Pendulum(states, t, horizon, actions)
        elif prob == "Pendulum_xyomega":
            return self.compute_cost_Pendulum_xy_omega(states, t, horizon, actions)
        if prob == "CartPoleContinuous":
            return self.compute_cost_CartPoleContinuous(states, t, horizon, actions)
        elif prob == "MountainCarContinuous":
            return self.compute_cost_MountainCarContinuous(states, t, horizon, actions)
        elif prob == "LunarLanderContinuous":
            return self.compute_cost_LunarLanderContinuous(states, t, horizon, actions)
        elif prob == "PandaReacher":
            return self.compute_cost_PandaReacher(states, t, horizon, actions, goal_state)
        elif prob == "PandaReacherDense":
            return self.compute_cost_PandaReacherDense(states, t, horizon, actions, goal_state)
        elif prob == "PandaPusher":
            return self.compute_cost_PandaPusher(states, t, horizon, actions, goal_state)
        elif prob == "MuJoCoReacher":
            return self.compute_cost_MuJoCoReacher(states, t, horizon, actions, goal_state)
        elif prob == "MuJoCoPusher":
            return self.compute_cost_MuJoCoPusher(states, t, horizon, actions, goal_state) 
        else:
            raise ValueError("Unknown problem")




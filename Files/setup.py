import numpy as np


class setup_class:
    def __init__(self, prob):
        self.memory = prob

        self.nb_actions = None

        # Generate random seeds
        self.random_seeds = [0, 8, 15]
        # print("random_seeds ", type(random_seeds[0]), "\n")
        self.nb_rep_episodes = len(self.random_seeds)

        # Constants
        self.batch_size = 32
        self.num_particles = 100
        self.quantiles = torch.tensor(np.array([0.0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]), dtype=torch.float32)
        self.num_quantiles = len(self.quantiles)
        self.nb_reps_MPC = 10
        self.nb_random = 0 # 10

        if prob == "CartPole":
            discrete = True
            horizon = 30
            max_episodes = 100
            max_steps = 200
            std = None
            change_prob = 0.01
            # change_prob = 0.05
            # change_prob = 0.1
            # change_prob = 0.3
            # change_prob = 0.5
            
            nb_actions = 2
            
            nb_top_particles = 5
            # nb_random = 10
            
            env = gym.make('CartPole-v0', render_mode="rgb_array").unwrapped # To save time since 200 instead of 500 steps per episode
            
            # Hyperparameters
            state_dim = env.observation_space.shape[0] # Since we only care about angle and omega which are given using env.state
            # action_dim = env.action_space.shape[0]  # For Pendulum, it's continuous
            action_dim = 1
            action_low = 0 #env.action_space.low[0]
            action_high = 1 # env.action_space.high[0]
            
            goal_state = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
            goal_state_dim = len(goal_state)

            states_low = torch.tensor([-4.8, -torch.inf, -0.41887903, -torch.inf])
            states_high = torch.tensor([4.8, torch.inf, 0.41887903, torch.inf])
            
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


        elif prob == "Acrobot":
            discrete = True
            horizon = 30
            max_episodes = 300
            max_steps = 200
            std = None
            change_prob = 0.01
            # change_prob = 0.05
            # change_prob = 0.1
            # change_prob = 0.3
            # change_prob = 0.5
            
            nb_actions = 3
            
            nb_top_particles = 5
            # nb_random = 10
            
            env = gym.make('Acrobot-v1', render_mode="rgb_array").unwrapped
            
            # Hyperparameters
            state_dim = env.observation_space.shape[0] # Since we only care about angle and omega which are given using env.state
            # action_dim = env.action_space.shape[0]  # For Pendulum, it's continuous
            action_dim = 1
            action_low = 0 #env.action_space.low[0]
            action_high = 2 # env.action_space.high[0]
            
            goal_state = torch.tensor([-1, 0, 1, 0, 0, 0], dtype=torch.float32)
            goal_state_dim = len(goal_state)

            states_low = torch.tensor([-1, -1, -1, -1, -12.566371, -28.274334])
            states_high = torch.tensor([1, 1, 1, 1, 12.566371, 28.274334])
            
            def compute_cost_Acrobot(states, t, horizon, actions):
                """
                Compute the cost based on the state and action for the CartPole environment.
                """

                # theta1 = np.arccos(np.clip(states[:, 0],-1,1))  # First joint angle
                # theta2 = np.arccos(np.clip(states[:, 2],-1,1))  # Second joint angle
                
                # distance_to_goal = 1 + (np.cos(theta1) + np.cos(theta1 + theta2))
                
                theta1 = torch.arccos(torch.clip(states[:, 0],-1,1))  # First joint angle
                theta2 = torch.arccos(torch.clip(states[:, 2],-1,1))  # Second joint angle
                
                distance_to_goal = 1 + (torch.cos(theta1) + torch.cos(theta1 + theta2))
                cost = distance_to_goal ** 2
                
                return cost
            
        elif prob == "LunarLander": # ToDo
            discrete = True
            horizon = 30
            max_episodes = 400
            # max_steps = 200 # No defined max episode length
            max_steps = 1000
            std = None
            change_prob = 0.01
            # change_prob = 0.05
            # change_prob = 0.1
            # change_prob = 0.3
            # change_prob = 0.5

            if change_prob == 0.01:
                std_string = "001"
            elif change_prob == 0.05:
                std_string = "005"
            elif change_prob == 0.1:
                std_string = "01"
            elif change_prob == 0.3:
                std_string = "03"
            elif change_prob == 0.5:
                std_string = "05"

            nb_actions = 3
            
            nb_top_particles = 5
            # nb_random = 10
            
            env = gym.make('LunarLander-v3', continuous=False, render_mode="rgb_array").unwrapped
            
            # Hyperparameters
            state_dim = env.observation_space.shape[0] # Since we only care about angle and omega which are given using env.state
            # action_dim = env.action_space.shape[0]  # For Pendulum, it's continuous
            action_dim = 1
            action_low = 0 #env.action_space.low[0]
            action_high = 2 # env.action_space.high[0]
            
            goal_state = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.float32)
            goal_state_dim = len(goal_state)

            states_low = torch.tensor([-2.5, -2.5, -10, -10, -6.2831855, -10, 0, 0])
            states_high = torch.tensor([2.5, 2.5, 10, 10, 6.2831855, 10, 1, 1])
            
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
                leftcontact = states[:, 6]
                rightcontact = states[:, 7]
                
                cost = x**2 + y**2 + vx**2 + vy**2 + theta**2 + omega**2
                # cost = distance_to_goal ** 2
                
                return cost

        elif prob == "MountainCar": # ToDo
            discrete = True
            horizon = 70
            # horizon = 100
            # max_episodes = 100
            max_episodes = 400
            max_steps = 200
            std = None
            # change_prob = 0.01
            # change_prob = 0.05
            change_prob = 0.1
            # change_prob = 0.3
            # change_prob = 0.5
            
            nb_top_particles = 5
            # nb_random = 10

            nb_actions = 3
            
            env = gym.make('MountainCar-v0', render_mode="rgb_array").unwrapped
            
            # Hyperparameters
            state_dim = env.observation_space.shape[0] # Since we only care about angle and omega which are given using env.state
            # action_dim = env.action_space.shape[0]  # For Pendulum, it's continuous
            action_dim = 1
            action_low = 0
            action_high = 3
            
            # goal_state = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
            goal_state = torch.tensor([0.5], dtype=torch.float32)
            goal_state_dim = len(goal_state)

            states_low = torch.tensor([-1.2, -0.07])
            states_high = torch.tensor([0.6, 0.07])
            
            def compute_cost_MountainCar(states, t, horizon, actions):
                goal_position = 0.5  # Position goal in Mountain Car
                gamma = 0.90 # 0.0  # Discount factor for delayed rewards
                
                # Distance reward: Encourage progress towards the goal
                distance_reward = (states[:, 0]-goal_position)**2 # -abs(goal_position - state[0])
                
                reverse_discount_factor = gamma**(horizon-t-1)
                
                return reverse_discount_factor*distance_reward 

        elif prob == "Pendulum_xyomega":
            discrete = False
            horizon = 15
            max_episodes = 300
            max_steps = 200

            # Current test values
            std = 0
            # std = 3e-1
            # std = 1.5

            # Older test values
            # std = 1e-1
            # std = 3e-1
            # std = 1
            # std = 1.5
            change_prob = None

            std_string = "0"
            # std_string = "3em1"
            # std_string = "15"
            
            nb_top_particles = 5
            # nb_random = 10
            
            env = gym.make('Pendulum-v1', render_mode="rgb_array").unwrapped
            
            # Hyperparameters
            state_dim = env.observation_space.shape[0]
            # state_dim = env.observation_space.shape[0]-1 # Since we only care about angle and omega which are given using env.state
            # action_dim = env.action_space.shape[0]  # For Pendulum, it's continuous
            action_dim = 1
            action_low = env.action_space.low[0]
            action_high = env.action_space.high[0]
            
            goal_state = torch.tensor([0, 0], dtype=torch.float32)
            goal_state_dim = len(goal_state)

            states_low = torch.tensor([-1, -1, -8])
            states_high = torch.tensor([1, 1, 8])
            
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
                theta = states[:, 0]#.detach().numpy()  # Shape: [num_particles]
                omega = states[:, 1]# .detach().numpy()  # Shape: [num_particles]

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

        elif prob == "Pendulum":
            discrete = False
            horizon = 15
            max_episodes = 300
            max_steps = 200

            # Current test values
            std = 0
            # std = 3e-1
            # std = 1.5

            # Older test values
            # std = 1e-1
            # std = 3e-1
            # std = 1
            # std = 1.5
            change_prob = None

            std_string = "0"
            
            nb_top_particles = 5
            # nb_random = 10
            
            env = gym.make('Pendulum-v1', render_mode="rgb_array").unwrapped
            
            # Hyperparameters
            state_dim = env.observation_space.shape[0]-1 # Since we only care about angle and omega which are given using env.state
            # action_dim = env.action_space.shape[0]  # For Pendulum, it's continuous
            action_dim = 1
            action_low = env.action_space.low[0]
            action_high = env.action_space.high[0]
            
            goal_state = torch.tensor([0, 0], dtype=torch.float32)
            goal_state_dim = len(goal_state)

            states_low = torch.tensor([-torch.inf, -torch.inf, -8])
            states_high = torch.tensor([torch.inf, torch.inf, 8])
            
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
            
        elif prob == "MountainCarContinuous":
            discrete = False
            horizon = 70
            # horizon = 100
            max_episodes = 100
            max_steps = 300 # 999 # in reality
            # std = 1e-2
            std = 1e-1
            # std = 3e-1
            # std = 6e-1
            change_prob = None
            
            nb_top_particles = 5
            # nb_random = 10
            
            env = gym.make('MountainCarContinuous-v0', render_mode="rgb_array").unwrapped
            
            # Hyperparameters
            state_dim = env.observation_space.shape[0] # Since we only care about angle and omega which are given using env.state
            # action_dim = env.action_space.shape[0]  # For Pendulum, it's continuous
            action_dim = 1
            action_low = env.action_space.low[0]
            action_high = env.action_space.high[0]
            
            # goal_state = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
            goal_state = torch.tensor([0.45], dtype=torch.float32)
            goal_state_dim = len(goal_state)

            states_low = torch.tensor([-1.2, -0.07])
            states_high = torch.tensor([0.6, 0.07])
            
            def compute_cost_MountainCarContinuous(states, t, horizon, actions):
                goal_position = 0.45  # Position goal in Mountain Car
                gamma = 0.90 # 0.0  # Discount factor for delayed rewards
                
                # Distance reward: Encourage progress towards the goal
                distance_reward = (states[:, 0]-goal_position)**2 # -abs(goal_position - state[0])
                
                reverse_discount_factor = gamma**(horizon-t-1)
                
                return reverse_discount_factor*distance_reward

        elif prob == "LunarLanderContinuous": # ToDo
            discrete = False
            horizon = 30
            max_episodes = 300
            max_steps = 200 # No defined max episode length
            std = 1e-1
            # std = 3e-1
            # std = 1
            # std = 1.5
            change_prob = None
            
            nb_top_particles = 5
            # nb_random = 10
            
            env = gym.make('LunarLander-v3', continuous=True, render_mode="rgb_array").unwrapped
            
            # Hyperparameters
            state_dim = env.observation_space.shape[0] # Since we only care about angle and omega which are given using env.state
            # action_dim = env.action_space.shape[0]  # For Pendulum, it's continuous
            action_dim = 2
            action_low = -1 #env.action_space.low[0]
            action_high = 1 # env.action_space.high[0]
            
            goal_state = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.float32)
            goal_state_dim = len(goal_state)

            states_low = torch.tensor([-2.5, -2.5, -10, -10, -6.2831855, -10, 0, 0])
            states_high = torch.tensor([2.5, 2.5, 10, 10, 6.2831855, 10, 1, 1])
            
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
                leftcontact = states[:, 6]
                rightcontact = states[:, 7]
                
                cost = x**2 + y**2 + vx**2 + vy**2 + theta**2 + omega**2
                # cost = distance_to_goal ** 2
                
                return cost

        elif prob == "PandaReacher":
            discrete = False
            horizon = 15
            max_episodes = 100
            # max_episodes = 400
            max_steps = 50 
            # std = 1e-1
            # std = 3e-1
            # std = 1
            std = 1.5
            change_prob = None

            if std == 1e-1:
                std_string = "1em1"
            elif std == 3e-1:
                std_string = "3em1"
            elif std == 1:
                std_string = "1"
            elif std == 1.5:
                std_string = "15"
            
            goal_state = None # Defined when resetting the env
            
            nb_top_particles = 5
            # nb_random = 10
            
            env = gym.make('PandaReach-v3', render_mode="rgb_array").unwrapped # Reward only when the end effector is at the goal position
            # env = gym.make('PandaReachDense-v3', render_mode='human').unwrapped # Reward at each time step based on the distance to the goal position
            
            # Hyperparameters    
            actions_low = env.action_space.low#[0] #[:3]
            actions_high = env.action_space.high#[0] #[:3]
            states_low = env.observation_space['observation'].low#[:3]
            states_high = env.observation_space['observation'].high#[:3]
            state_dim = len(states_low)
            action_dim = len(actions_low)
            action_low = actions_low[0]
            action_high = actions_high[0]
            goal_state_dim = 3 #len(states_low)
            
            states_low = torch.tensor([-10, -10, -10, -10, -10, -10])
            states_high = torch.tensor([10, 10, 10, 10, 10, 10])
            
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

        elif prob == "PandaPusher": # ToDo
            discrete = False
            horizon = 15
            max_episodes = 100
            max_steps = 50 
            std = 1e-1
            # std = 3e-1
            # std = 1
            # std = 1.5
            change_prob = None
            
            std_string = "1em1"
            
            goal_state = None # Defined when resetting the env
            
            nb_top_particles = 5
            # nb_random = 10
            
            # env = gym.make("PandaPush-v3").unwrapped # Reward only when the end effector is at the goal position
            env = gym.make('PandaPushDense-v3', render_mode="rgb_array").unwrapped # Reward at each time step based on the distance to the goal position
            
            # Hyperparameters    
            actions_low = env.action_space.low #[:3]
            actions_high = env.action_space.high #[:3]
            states_low = env.observation_space['observation'].low #[:3]
            states_high = env.observation_space['observation'].high #[:3]
            state_dim = len(states_low)
            action_dim = len(actions_low)
            action_low = actions_low[0]
            action_high = actions_high[0]
            goal_state_dim = 3 #len(states_low)

            states_low = torch.tensor([-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10])
            states_high = torch.tensor([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
            
            def compute_cost_PandaPusher(states, t, horizon, actions, goal_state=None):
                # print("goal_state ", goal_state, "\n")
                # print("states[0] ", states[0], "\n")
                # print("states.shape ", states.shape, "\n")
                # print("states[:, :3].shape ", states[:, :3].shape, "\n")
                # print("states[:, 6:9].shape ", states[:, 6:9].shape, "\n")
                
                goal_state = torch.tensor(goal_state, dtype=torch.float32, device=states.device).reshape(1, 3)
                
                costs = torch.norm(states[:, :3]-states[:, 6:9], dim=1)+torch.norm(states[:, 6:9]-goal_state, dim=1)
                
                # print("cost1 ", torch.norm(states[0, :3]-states[0, 6:9])+torch.norm(states[0, 6:9]-goal_state), '\n')
                # print("cost2 ", torch.norm(states[1, :3]-states[1, 6:9])+torch.norm(states[1, 6:9]-goal_state), '\n')
                # print("costs ", costs, "\n")
                
                return costs
            

        elif prob == "MuJoCoReacher":
            discrete = False
            horizon = 15
            max_episodes = 400
            max_steps = 50 
            std = 1e-1
            # std = 3e-1
            # std = 1
            # std = 1.5
            change_prob = None

            if std == 1e-1:
                std_string = "1em1"
            elif std == 3e-1:
                std_string = "3em1"
            elif std == 1:
                std_string = "1"
            elif std == 1.5:
                std_string = "15"
            
            
            goal_state = None # Defined when resetting the env
            
            nb_top_particles = 5
            # nb_random = 10
            
            env = gym.make('Reacher-v5', render_mode="rgb_array").unwrapped
            
            # Hyperparameters    
            actions_low = env.action_space.low#[:3]
            actions_high = env.action_space.high#[:3]
            # states_low = env.observation_space['observation'].low#[:3]
            # states_high = env.observation_space['observation'].high#[:3]
            state_dim = 8
            action_dim = len(actions_low)
            
            action_low = actions_low[0]
            action_high = actions_high[0]
            
            goal_state_dim = 2

            states_low = torch.tensor([-1, -1, -1, -1, -100, -100, -torch.inf, -torch.inf])
            states_high = torch.tensor([1, 1, 1, 1, 100, 100, torch.inf, torch.inf])
            
            def compute_cost_MuJoCoReacher(states, t, horizon, actions, goal_state=None):
                # cost1 = torch.sqrt(states[0, -2]**2+states[0, -1]**2)+0.1*(torch.norm(actions[0]))**2
                # costs2 = torch.sqrt(states[1, -2]**2+states[1, -1]**2)+0.1*(torch.norm(actions[1]))**2
                costs = torch.sqrt(states[:, -2]**2+states[:, -1]**2)+0.1*(torch.norm(actions, dim=1))**2
                # print("cost1 ", cost1, "\n")
                # print("costs2 ", costs2, "\n")
                # print("costs ", costs, "\n")
                return costs
            # torch.norm(states[:, :3]-torch.tensor(goal_state, dtype=torch.float32))


        elif prob == "MuJoCoPusher": # ToDo 
        discrete = False
        horizon = 15
        max_episodes = 100
        max_steps = 50 
        std = 1e-1
        # std = 3e-1
        # std = 1
        # std = 1.5
        change_prob = None
        
        goal_state = None # Defined when resetting the env
        
        nb_top_particles = 5
        # nb_random = 10
        
        env = gym.make('Pusher-v5', render_mode="rgb_array").unwrapped
        
        # Hyperparameters    
        actions_lows = env.action_space.low#[:3]
        actions_highs = env.action_space.high#[:3]
        states_low = env.observation_space['observation'].low#[:3]
        states_high = env.observation_space['observation'].high#[:3]
        state_dim = len(states_low)
        action_dim = len(actions_lows)
        
        action_low = actions_lows[0]
        action_high = actions_highs[0]
        
        goal_state_dim = 3

        # TBD
        states_low = None # torch.tensor([-1, -1, -1, -1, -100, -100, -torch.inf, -torch.inf])
        states_high = None # torch.tensor([1, 1, 1, 1, 100, 100, torch.inf, torch.inf])
        
        def compute_cost_MuJoCoPusher(states, t, horizon, actions, goal_state=None):
            return torch.norm(states[:, 14:17]-states[:, 17:20], dim=1)+torch.norm(states[:, 17:20]-states[:, 20:], dim=1)
            
            # return torch.sqrt(states[:, -2]**2+states[:, -1]**2)+0.1*(torch.norm(actions))**2
        
        # torch.norm(states[:, :3]-torch.tensor(goal_state, dtype=torch.float32))

    def compute_cost(self, prob, states, t, horizon, actions, goal_state=None):
        if prob == "CartPole":
            return self.compute_cost_CartPole(states, t, horizon, actions)
        elif prob == "Acrobot":
            return self.compute_cost_Acrobot(states, t, horizon, actions)
        elif prob == "LunarLander":
            return self.compute_cost_LunarLander(states, t, horizon, actions)
        elif prob == "MountainCar":
            return self.compute_cost_MountainCar(states, t, horizon, actions)
        elif prob == "Pendulum":
            return self.compute_cost_Pendulum(states, t, horizon, actions)
        elif prob == "Pendulum_xyomega":
            return self.compute_cost_Pendulum_xy_omega(states, t, horizon, actions)
        elif prob == "MountainCarContinuous":
            return self.compute_cost_MountainCarContinuous(states, t, horizon, actions)
        elif prob == "LunarLanderContinuous":
            return self.compute_cost_LunarLanderContinuous(states, t, horizon, actions)
        elif prob == "PandaReacher":
            return self.compute_cost_PandaReacher(states, t, horizon, actions, goal_state)
        elif prob == "PandaPusher":
            return self.compute_cost_PandaPusher(states, t, horizon, actions, goal_state)
        elif prob == "MuJoCoReacher":
            return self.compute_cost_MuJoCoReacher(states, t, horizon, actions, goal_state)
        elif prob == "MuJoCoPusher":
            return self.compute_cost_MuJoCoPusher(states, t, horizon, actions, goal_state) 
        else:
            raise ValueError("Unknown problem")




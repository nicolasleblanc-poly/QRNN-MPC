import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress UserWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Suppress DeprecationWarnings
from evotorch import Problem, Solution, SolutionBatch
from evotorch.algorithms import CEM
from evotorch.logging import StdOutLogger
import torch
import numpy as np
import random

import logging

logging.getLogger("evotorch").disabled = True
logging.getLogger("evotorch").setLevel(logging.ERROR)  # or logging.CRITICAL

from state_pred_models import quantile_loss, quantile_loss_median, mse_loss

# Ideas and code taken from: https://docs.evotorch.ai/v0.1.1/examples/notebooks/reacher_mpc/#definitions




class PlanningProblem(Problem):
    def __init__(self, prob_vars, state, model_state, use_sampling, use_mid):
        
        self.prob = prob_vars.prob
        self.action_dim = prob_vars.action_dim
        self.horizon = prob_vars.horizon
        # self.state = np.asarray(state, dtype="float32")
        self.state_dim = prob_vars.state_dim
        self.state = torch.tensor(state, dtype=torch.float32).reshape(self.state_dim)
        # self.state_dim = self.state.shape[0]
        # self.state = torch.as_tensor(reacher_state(self.observation), dtype=torch.float32)
        self.action_low = prob_vars.action_low
        self.action_high = prob_vars.action_high
        self.states_low = prob_vars.states_low
        self.states_high = prob_vars.states_high
        self.goal_state = torch.tensor(prob_vars.goal_state, dtype=torch.float32)
        self.model_state = model_state
        self.use_sampling = use_sampling
        self.use_mid = use_mid
        self.compute_cost = prob_vars.compute_cost

        # print("self.state.shape ", self.state.shape, "\n")
        # print("self.state ", self.state, "\n")

        # print("horizon ", horizon, "\n")

        super().__init__(
            objective_sense="min",
            initial_bounds=(-0.0001, 0.0001),
            solution_length=(prob_vars.horizon * prob_vars.action_dim),
        )

    @torch.no_grad()
    def predict_plan_outcome(self, state, state_dim, action_dim, plan_batch, model_state, use_sampling, use_mid, action_low, action_high, states_low, states_high):

        # print("plan_batch.shape ", plan_batch.shape, "\n")
        # print("plan_batch.size ", plan_batch.size, "\n")

        # num_particles, plan_length = plan_batch.size
        num_particles = plan_batch.shape[0]
        # print("num_particles ", num_particles, "\n")

        # print("state.shape ", state.shape, "\n")
        
        # state_batch = state*torch.ones(num_particles, len(state))
        state_batch = state.repeat(num_particles, 1)  # Correct way to repeat along the first dimension

        # print("state_batch.shape ", state_batch.shape, "\n")

        plan_batch = plan_batch.reshape(num_particles, -1, action_dim)
        # print("plan_batch.shape 1 ", plan_batch.shape, "\n")
        horizon = plan_batch.shape[1]
        # print("horizon ", horizon, "\n")

        # cost = torch.zeros(num_particles, dtype=torch.float32)
        
        for t in range(horizon):
            # if self.action_dim > 1:
            #     action_batch = plan_batch[:, t:t+action_dim, :]
            # else:
            #     action_batch = plan_batch[:, t, :]
            action_batch = plan_batch[:, t, :]
            # state_batch = model_QRNN(state_batch, action_batch)

            # print("state_batch.shape 2 ", state_batch.shape, "\n")
            # print("action_batch.shape 2 ", action_batch.shape, "\n")

            state_batch = torch.clip(state_batch, states_low, states_high)
            action_batch = torch.clip(action_batch, action_low, action_high)

            state_batch = model_state(state_batch, action_batch)

            costs = self.compute_cost(self.prob, state_batch, self.horizon, self.horizon, action_batch, self.goal_state)
            
        return state_batch
        # return costs

    def _evaluate_batch(self, solutions: SolutionBatch):

        # costs = self.predict_plan_outcome(self.state, self.state_dim, self.action_dim, solutions.values, self.model_QRNN, self.use_sampling, self.use_mid, self.action_low, self.action_high, self.states_low, self.states_high)

        final_states = self.predict_plan_outcome(self.state, self.state_dim, self.action_dim, solutions.values, self.model_state, self.use_sampling, self.use_mid, self.action_low, self.action_high, self.states_low, self.states_high)
        
        final_xys = final_states[:, -self.state_dim:]

        # print("final_xys.shape ", final_xys.shape, "\n")
        # print("solutions.values.shape ", solutions.values.shape, "\n")
        # print("self.action_dim ", self.action_dim, "\n")
        # print("solutions.values[:, -self.action_dim] ", solutions.values[:, -self.action_dim:], "\n")

        if self.action_dim > 1:
            errors = self.compute_cost(self.prob, final_xys, self.horizon, self.horizon, solutions.values[:, -self.action_dim:], self.goal_state)
        else:
            errors = self.compute_cost(self.prob, final_xys, self.horizon, self.horizon, solutions.values[:, -self.action_dim], self.goal_state)
            
        # torch.linalg.norm(final_xys - self.target_xy, dim=-1)
        solutions.set_evals(errors)

        # solutions.set_evals(costs)

def do_planning_cem(prob_vars, state, model_state, use_sampling, use_mid):
    problem = PlanningProblem(prob_vars, state, model_state, use_sampling, use_mid)
    searcher = CEM(
        problem,
        stdev_init = 0.5,
        popsize = 250,  # population size
        parenthood_ratio = 0.5,
        stdev_max_change = 0.2,
    )
    searcher.run(20)  # run for this many generations
    return searcher.status["best"].values[:prob_vars.action_dim].clone().numpy()

def start_50NN_MSENN_MPC_CEM(prob_vars, env, seed, model_state, replay_buffer_state, optimizer_state, loss_state, use_sampling, use_mid):
    
    episode_reward_list = []
    episode_success_rate = [] # For Panda Gym envs

    nb_episode_success = 0 # For Panda Gym envs

    # if prob == "PandaReacher" or prob == "PandaPusher":
    #     states_low = torch.tensor(-10)
    #     states_high = torch.tensor(10)

    # if prob == "MuJoCoReacher":
    #     states_low = torch.tensor([-1, -1, -1, -1, -100, -100, -torch.inf, -torch.inf])
    #     states_high = torch.tensor([1, 1, 1, 1, 100, 100, torch.inf, torch.inf])

    for episode in range(prob_vars.max_episodes):
        state, _ = env.reset(seed=seed)
        episode_reward = 0
        # is_success_bool = False # For Panda Gym envs
        done = False
        actions_list = []
        if prob_vars.prob == "Pendulum":
            state = env.state.copy()
        if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "PandaReacherDense":
            prob_vars.goal_state = state['desired_goal'] # 3 components for Reach and for Push
            state = state['observation'] #[:3] # 6 components for Reach, 18 components for Push
            # print("goal_state ", goal_state, "\n")
        if prob_vars.prob == "MuJoCoReacher":
            prob_vars.goal_state = np.array([state[4], state[5]])
            state = np.array([state[0], state[1], state[2], state[3], state[6], state[7], state[8], state[9]])
        if prob_vars.prob == "MuJoCoPusher":
            prob_vars.goal_state = np.array([state[20], state[21], state[22]])
        
        costs = []
        episodic_step_rewards = []
        episodic_step_reward_values = []
        
        # for step in tqdm(range(max_steps)):
        for step in range(prob_vars.max_steps):
            # Get the current state
            # """ Need to change this!!!!!!!!!!!!!!!!! """
            # state = env.state
            # print("step ", step, "\n")
            if prob_vars.prob == "Pendulum":
                state = env.state.copy()
            
            # particles = np.clip(particles, action_low, action_high)
            
            # best_particle, action, best_cost, particles = choose_action(prob, state, horizon, particles, do_RS, use_sampling, use_mid, use_ASGNN, model_QRNN, model_ASN, action_dim, action_low, action_high, nb_reps_MPC, std, change_prob, nb_top_particles, nb_random, episode=episode, step=step, goal_state=goal_state)
            # best_particle, particles, cost = particle_filtering_cheating(particles, env, state, horizon, nb_reps=5, using_Env=usingEnv, episode=episode, step=step)
            
            action = do_planning_cem(prob_vars, state, model_state, use_sampling, use_mid)

            # print("action ", action, "\n")
            
            # costs.append(best_cost)
            
            # print("best_particle ", best_particle, "\n")
            if prob_vars.prob == "Pendulum" or prob_vars.prob == "MountainCarContinuous" or prob_vars.prob == "Pendulum_xyomega":
                # action = [best_particle[0]]
                action = [action]
                actions_list.append(list(action))
            
            # elif prob == "PanadaReacher" or prob == "MuJoCoReacher" or prob == "PandaPusher" or prob == "MuJoCoPusher" or prob == "LunarLanderContinuous":
                # action = best_particle[:action_dim]
                # actions_list.append(list(action))
            
            elif prob_vars.prob == "CartPole" or prob_vars.prob == "Acrobot" or prob_vars.prob == "MountainCar" or prob_vars.prob == "LunarLander":
                action = int(action)
                actions_list.append(action)
            
            # actions_list.append(action)
            # actions_list.append(list(action))
            
            # print("env.step(env.action_space.sample()) ", env.step(env.action_space.sample()), "\n")
            # print("action.shape ", action.shape, "\n")
            # print("env.action_space.sample().shape ", env.action_space.sample().shape, "\n")
            
            # Apply the first action from the optimized sequence
            next_state, reward, done, terminated, info = env.step(action)
            
            episode_reward += reward
            # actions_list.append(action)
            # actions_list.append(list(action))
            
            # Apply the first action from the optimized sequence
            # next_state, reward, done, terminated, info = env.step(action)
            # episode_reward += reward
            if prob_vars.prob == "Pendulum":
                # state = env.state.copy()
                next_state = env.state.copy()

            if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "PandaReacherDense":
                goal_state = next_state['desired_goal'] # 3 components
                next_state = next_state['observation']#[:3] # 6 components for Reacher, 18 components for Pusher
                # is_success_bool = info['is_success']
                
            if prob_vars.prob == "MuJoCoReacher":
                next_state = np.array([next_state[0], next_state[1], next_state[2], next_state[3], next_state[6], next_state[7], next_state[8], next_state[9]])
            
            next_state = next_state.reshape(prob_vars.state_dim)

            # print("state ", state, "next_state ", next_state, "\n")
            # print("states[0] ", state[0], "states[1] ", state[1], "\n")
            
            # episodic_step_rewards.append(episode_reward)
            # episodic_step_reward_values.append(reward)
            
            # next_state = env.state.copy()
            # Store experience in replay buffer
            # print("state ", state, "\n")
            
            # if prob != "CartPole" and prob != "Acrobot":
            #     replay_buffer_QRNN.append((state, action, reward, next_state, done))
            # else:
            #     replay_buffer_QRNN.append((state, np.array([action]), reward, next_state, terminated))
            if prob_vars.prob == "CartPole" or prob_vars.prob == "Acrobot" or prob_vars.prob == "MountainCar" or prob_vars.prob == "LunarLander":
                replay_buffer_state.append((state, np.array([action]), reward, next_state, terminated))
            else:
                state = state.reshape(prob_vars.state_dim)
                next_state = next_state.reshape(prob_vars.state_dim)
                # print("state ", state, "\n")
                # print("next_state ", next_state, "\n")
                replay_buffer_state.append((state, action, reward, next_state, terminated))
            
            if len(replay_buffer_state) < prob_vars.batch_size:
                pass
            else:
                batch = random.sample(replay_buffer_state, prob_vars.batch_size)
                states, actions_train, rewards, next_states, dones = zip(*batch)
                # print("batch states ", states, "\n")
                # print("type(states[0]) ", type(states[0]), "\n")
                states = torch.tensor(states, dtype=torch.float32)
                actions_tensor = torch.tensor(actions_train, dtype=torch.float32).reshape(prob_vars.batch_size, prob_vars.action_dim)
                # print("actions.shape ", actions_tensor, "\n")
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                # print("states.shape ", states.shape, "\n")
                # print("actions_tensor.shape ", actions_tensor.shape, "\n")


                if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaReacherDense":
                    # Clip states to ensure they are within the valid range
                    # before inputting them to the model (sorta like normalization)
                    states = torch.clip(states, prob_vars.states_low, prob_vars.states_high)
                
                # Predict next state quantiles
                # predicted_quantiles = model_QRNN(states, actions_tensor)  # Shape: (batch_size, num_quantiles, state_dim)
                
                # Use next state as target (can be improved with target policy)
                # target_quantiles = next_states

                predicted_next_states = model_state(states, actions_tensor)
                
                # Compute the target quantiles (e.g., replicate next state across the quantile dimension)
                # target_quantiles = next_states.unsqueeze(-1).repeat(1, 1, num_quantiles)

                # Compute Quantile Huber Loss
                # loss = quantile_loss(predicted_quantiles, target_quantiles, prob_vars.quantiles)

                loss = loss_state(predicted_next_states, next_states)
                
                # Optimize the model_QRNN
                optimizer_state.zero_grad()
                loss.backward()
                optimizer_state.step()
            
            if prob_vars.prob == "MuJoCoReacher":
                if np.sqrt(next_state[-2]**2+next_state[-1]**2) < 0.05:
                    # print("Reached target position \n")
                    done = True
            
            done = done or terminated
            if done:
                nb_episode_success += 1
                break
            
            state = np.copy(next_state).reshape(prob_vars.state_dim)
            
            # Shift all particles to the left by removing the first element
            # particles[:, :-action_dim] = particles[:, action_dim:]
            # The last column will be determined in MPC
            
            # particles = np.clip(particles, action_low, action_high)
            
        # print("best_particle ", best_particle, "\n")
        # print("actions ", actions, "\n")
        # print('horizon: %d, episode: %d, reward: %d' % (horizon, episode, episode_reward))
        # episode_reward_list.append(episode_reward)
        # episode_reward_list_MPC_PF_QRNN_WithActionSequenceGeneratorNN.append(episode_reward)
        
        # episode_reward_list_withASGNN[episode] = episode_reward
        episode_reward_list.append(episode_reward)

        episode_success_rate.append(nb_episode_success/(episode+1)) # Episodic success rate for Panda Gym envs
        # episode_success_rate_withASGNN.append(nb_episode_success) # /max_steps # Episodic success rate for Panda Gym envs
        
        # print(f'episode: {episode}, reward: {episode_reward}')
        # # episode_reward_list.append(episode_reward)
        # print("actions_list ", actions_list, "\n")
        
        """ Print stuff """
        # if prob == 'PandaReacher':
        #     print(f'episode: {episode}, reward: {episode_reward}')
        #     print("nb_episode_success ", nb_episode_success, "\n")
        #     print("np.linalg.norm(goal_state-state) ", np.linalg.norm(goal_state-next_state[:3]), "\n")
        #     print("episode", episode, "actions_list ", actions_list, "\n")
            
        # if prob == "MuJoCoReacher":
        #     print("np.linalg.norm(goal_state-state)=np.sqrt(next_state[-2]**2+next_state[-1]**2) ", np.sqrt(next_state[-2]**2+next_state[-1]**2), "\n")

    if prob_vars.prob == "PandaReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaReacherDense":
        return episode_reward_list, episode_success_rate
    else:
        return episode_reward_list



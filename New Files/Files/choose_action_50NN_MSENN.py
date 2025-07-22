import numpy as np
import random
import torch

# from mpc import mpc_func
from mpc_50NN_MSENN import mpc_50NN_MSENN_func
from particle_filtering import particle_filtering_func, discrete_cem_func, continuous_cem_func

def choose_action_func_50NN_MSENN(prob_vars, state, particles, do_RS, use_sampling, use_mid, use_ASGNN, model_state, model_ASN, episode=0, step=1, goal_state=None):

    best_cost = float('inf')
    best_action_sequence = None

    nb_reps = prob_vars.nb_reps_MPC


    if do_RS:
        nb_reps = 1

    for rep in range(nb_reps):
    
        sim_states = torch.tensor(state, dtype=torch.float32).repeat(prob_vars.num_particles, 1)


        costs = mpc_50NN_MSENN_func(prob_vars, sim_states, particles, use_ASGNN, model_state, use_sampling, use_mid, model_ASN)

        min_idx = torch.argmin(costs)
        
        
        if costs[min_idx] < best_cost:
            best_cost = costs[min_idx]
            best_action_sequence = particles[min_idx].copy()

        if best_cost == None or best_action_sequence is None: # For debugging
            best_cost = costs[0]
            best_action_sequence = particles[0].copy()
            print("costs ", costs, "\n")
            print("best_action_sequence ", best_action_sequence, "\n")
            print("sim_states ", sim_states, "\n")
            print("state ", state, "\n")
            print("particles ", particles, "\n")

        if not do_RS:
            if prob_vars.use_CEM:
                if prob_vars.discrete:
                    best_first_action, particles = discrete_cem_func(prob_vars, particles, costs, best_action_sequence)
                else:
                    best_first_action, particles = continuous_cem_func(prob_vars, particles, costs, best_action_sequence)

            else:
                best_first_action, particles = particle_filtering_func(prob_vars, particles, costs, best_action_sequence)

        else:
            if prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "LunarLanderContinuous" or prob_vars.prob == "PandaReacherDense":
                
                particles[prob_vars.num_particles-prob_vars.nb_random:] = np.random.uniform(prob_vars.action_low, prob_vars.action_high, (prob_vars.nb_random, prob_vars.horizon*prob_vars.action_dim))
                best_first_action = best_action_sequence[:prob_vars.action_dim] # .item()
            else: # Pendulum, MountainCarContinuous
                particles[prob_vars.num_particles-prob_vars.nb_random:] = np.random.uniform(prob_vars.action_low, prob_vars.action_high, (prob_vars.nb_random, prob_vars.horizon))
                best_first_action = best_action_sequence[0].item()

    return best_action_sequence, best_first_action, best_cost, particles


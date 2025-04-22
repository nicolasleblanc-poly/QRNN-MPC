import numpy as np
import random
import torch

# from mpc import mpc_func
from mpc_50NN_MSENN import mpc_50NN_MSENN_func
from particle_filtering import particle_filtering_func

def choose_action_func_50NN_MSENN(prob_vars, state, particles, do_RS, use_sampling, use_mid, use_ASGNN, model_QRNN, model_ASN, episode=0, step=1, goal_state=None):

    best_cost = float('inf')
    best_action_sequence = None

    nb_reps = prob_vars.nb_reps_MPC
    # horizon = prob_vars.horizon
    # action_dim = prob_vars.action_dim
    # action_low = prob_vars.action_low
    # action_high = prob_vars.action_high
    # states_low = prob_vars.states_low
    # states_high = prob_vars.states_high
    # std = prob_vars.std
    # change_prob = prob_vars.change_prob
    # nb_top_particles = prob_vars.nb_top_particles
    # nb_random = prob_vars.nb_random

    # if prob == "PandaReacher" or prob == "PandaPusher":
    #     states_low = torch.tensor(-10)
    #     states_high = torch.tensor(10)

    # if prob == "MuJoCoReacher":
    #     states_low = torch.tensor([-1, -1, -1, -1, -100, -100, -torch.inf, -torch.inf])
    #     states_high = torch.tensor([1, 1, 1, 1, 100, 100, torch.inf, torch.inf])

    if do_RS:
        nb_reps = 1

    for rep in range(nb_reps):
    
        # costs, best_action_sequence = objective_function(particles, sim_states, horizon, model)
        # num_particles = particles.shape[0]
        
        
        sim_states = torch.tensor(state, dtype=torch.float32).repeat(prob_vars.num_particles, 1)

        # best_cost = float('inf')
        # best_action_sequence = None

        costs = mpc_50NN_MSENN_func(prob_vars, sim_states, particles, use_ASGNN, model_QRNN, use_sampling, use_mid, model_ASN)

        min_idx = torch.argmin(costs)
        
        # print("costs[min_idx] ", costs[min_idx], "\n")
        # print("best_cost ", best_cost, "\n")
        
        if costs[min_idx] < best_cost:
            best_cost = costs[min_idx]
            best_action_sequence = particles[min_idx].copy()

        if best_cost == None or best_action_sequence is None:
            best_cost = costs[0]
            best_action_sequence = particles[0].copy()
            print("costs ", costs, "\n")
            print("best_action_sequence ", best_action_sequence, "\n")
            print("sim_states ", sim_states, "\n")
            print("state ", state, "\n")
            print("particles ", particles, "\n")
        
        # if prob == "Acrobot" : # or prob == "CartPole"
        #     min_idx = torch.argmin(costs)
        #     if costs[min_idx] < best_cost:
        #         best_cost = costs[min_idx]
        #         best_action_sequence = particles[min_idx].copy()

        # print("particles[0] ", particles[0], "\n")
        # print("best_action_sequence ", best_action_sequence, "\n")

        particles[0] = best_action_sequence

        best_first_action, particles = particle_filtering_func(prob_vars, particles, costs, best_action_sequence)   

    # best_first_action = int(best_action_sequence[0].item())
    return best_action_sequence, best_first_action, best_cost, particles


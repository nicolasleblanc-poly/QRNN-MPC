import numpy as np
import random
import torch

from mpc import mpc_func

def particle_filtering_func(prob_vars, particles, costs, best_action_sequence):

    # print("costs.shape ", costs.shape, "\n")
    # if not prob_vars.discrete:
    #     costs = costs.sum(dim=1)
    # print("costs.shape ", costs.shape, "\n")

    # Get the indices of the top 15 particles
    top_indices = torch.argsort(costs)[:prob_vars.nb_top_particles]
    top_particles = particles[top_indices]

    # Randomly select len(particles)-nb_random-1 particles from the top nb_top_particles with replacement
    sampled_indices = np.random.choice(prob_vars.nb_top_particles, prob_vars.num_particles-prob_vars.nb_random-1, replace=True)
    sampled_particles = top_particles[sampled_indices]

    # # Add Gaussian noise to the sampled particles
    # noise = np.random.normal(0, std, sampled_particles.shape)
    # # particles[1:len(particles)-nb_random] = np.clip(sampled_particles + noise, action_low, action_high)

    if prob_vars.prob == "CartPole":
        # Randomly change some of the sampled particles
        change_indices = np.random.choice([True, False], len(sampled_particles), p=[prob_vars.change_prob, 1-prob_vars.change_prob])
        sampled_particles[change_indices] = np.random.randint(0, 2, (len(sampled_particles[change_indices]), prob_vars.horizon))
        
        particles[1:prob_vars.num_particles-prob_vars.nb_random] = sampled_particles
        # particles[1:len(particles)-nb_random] = np.clip(sampled_particles + noise, action_low, action_high)
        
        # Randomly initialize the remaining particles
        particles[prob_vars.num_particles-prob_vars.nb_random:] = np.random.randint(0, 2, (prob_vars.nb_random, prob_vars.horizon))
        best_first_action = int(best_action_sequence[0].item())
        
    elif prob_vars.prob == "Acrobot" or prob_vars.prob == "MountainCar":
        # Randomly change some of the sampled particles
        change_indices = np.random.choice([True, False], len(sampled_particles), p=[prob_vars.change_prob, 1-prob_vars.change_prob])
        sampled_particles[change_indices] = np.random.randint(0, 3, (len(sampled_particles[change_indices]), prob_vars.horizon))
        
        particles[1:prob_vars.num_particles-prob_vars.nb_random] = sampled_particles
        # particles[1:len(particles)-nb_random] = np.clip(best_action_sequence + noise, action_low, action_high)
        
        # Randomly initialize the remaining particles
        particles[prob_vars.num_particles-prob_vars.nb_random:] = np.random.randint(0, 3, (prob_vars.nb_random, prob_vars.horizon))
        best_first_action = int(best_action_sequence[0].item())
    
    elif prob_vars.prob == "LunarLander":
        # Randomly change some of the sampled particles
        change_indices = np.random.choice([True, False], len(sampled_particles), p=[prob_vars.change_prob, 1-prob_vars.change_prob])
        sampled_particles[change_indices] = np.random.randint(0, 4, (len(sampled_particles[change_indices]), prob_vars.horizon))
        
        particles[1:prob_vars.num_particles-prob_vars.nb_random] = sampled_particles
        # particles[1:len(particles)-nb_random] = np.clip(best_action_sequence + noise, action_low, action_high)
        
        # Randomly initialize the remaining particles
        particles[prob_vars.num_particles-prob_vars.nb_random:] = np.random.randint(0, 4, (prob_vars.nb_random, prob_vars.horizon))
        best_first_action = int(best_action_sequence[0].item())

    else: # Pendulum, MountainCarContinuous, PandaReacher, MuJoCoReacher
        # Add Gaussian noise to the sampled particles
        noise = np.random.normal(0, prob_vars.std, sampled_particles.shape)
        # particles[1:len(particles)-nb_random] = np.clip(sampled_particles + noise, action_low, action_high)
        
        particles[1:prob_vars.num_particles-prob_vars.nb_random] = np.clip(sampled_particles + noise, prob_vars.action_low, prob_vars.action_high)
        # Randomly initialize the remaining particles
        # print("len(particles) ", len(particles), "\n")
        # print("nb_random ", nb_random, "\n")
        
        # print("(nb_random, horizon*action_dim) ", (nb_random, horizon*action_dim), "\n")
        # print("np.random.uniform(actions_low, actions_high, (nb_random, horizon*action_dim)).shape ", np.random.uniform(action_low, action_high, (nb_random, horizon*action_dim)).shape, "\n")
        # print("particles[len(particles)-nb_random:].shape ", particles[len(particles)-nb_random:].shape, "\n")
        
        if prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "LunarLanderContinuous" or prob_vars.prob == "PandaReacherDense":
            # print("best_cost ", best_cost, "\n")
            # print("best_action_sequence ", best_action_sequence, "\n")
            
            particles[prob_vars.num_particles-prob_vars.nb_random:] = np.random.uniform(prob_vars.action_low, prob_vars.action_high, (prob_vars.nb_random, prob_vars.horizon*prob_vars.action_dim))
            best_first_action = best_action_sequence[:prob_vars.action_dim] # .item()
        else: # Pendulum, MountainCarContinuous
            particles[prob_vars.num_particles-prob_vars.nb_random:] = np.random.uniform(prob_vars.action_low, prob_vars.action_high, (prob_vars.nb_random, prob_vars.horizon))
            best_first_action = best_action_sequence[0].item()
    
    return best_first_action, particles


def discrete_cem_func(prob_vars, particles, costs, best_action_sequence):

    # costs = costs.sum(dim=1)
    # if not prob_vars.discrete:
    #     costs = costs.sum(dim=1)

    # num_sequences, sequence_length = particles.shape
    num_particles = prob_vars.num_particles
    horizon = prob_vars.horizon
    num_actions = prob_vars.nb_actions

    laplace_alpha = prob_vars.laplace_alpha

    # Get the indices of the top 15 particles
    top_indices = torch.argsort(costs)[:prob_vars.nb_top_particles]
    top_particles = particles[top_indices]

    position_probs = np.zeros((horizon, num_actions))

    for t in range(horizon):
        counts = np.bincount(top_particles[:, t].flatten(), minlength=num_actions)
        smoothed_counts = counts + laplace_alpha
        position_probs[t] = smoothed_counts / smoothed_counts.sum()

    new_particles = np.zeros((prob_vars.num_particles, horizon), dtype=int)
    
    for t in range(horizon):
        new_particles[:, t] = np.random.choice(num_actions, size=num_particles, p=position_probs[t])

    if prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "LunarLanderContinuous" or prob_vars.prob == "PandaReacherDense":
        # print("best_cost ", best_cost, "\n")
        # print("best_action_sequence ", best_action_sequence, "\n")
        
        particles[prob_vars.num_particles-prob_vars.nb_random:] = np.random.uniform(prob_vars.action_low, prob_vars.action_high, (prob_vars.nb_random, prob_vars.horizon*prob_vars.action_dim))
        best_first_action = best_action_sequence[:prob_vars.action_dim] # .item()
    else: # Pendulum, MountainCarContinuous
        particles[prob_vars.num_particles-prob_vars.nb_random:] = np.random.uniform(prob_vars.action_low, prob_vars.action_high, (prob_vars.nb_random, prob_vars.horizon))
        best_first_action = best_action_sequence[0].item()

    return best_first_action, new_particles

def continuous_cem_func(prob_vars, particles, costs, best_action_sequence, noisy=False):

    # costs = costs.sum(dim=1)

    # num_sequences, sequence_length = particles.shape
    num_particles = prob_vars.num_particles
    horizon = prob_vars.horizon

    # Get the indices of the top 15 particles
    top_indices = torch.argsort(costs)[:prob_vars.nb_top_particles]
    top_particles = particles[top_indices]

    mu = np.mean(top_particles, axis=0)

    if prob_vars.action_dim > 1:
        sigma = np.cov(top_particles, rowvar=False, bias = True)

    else: # 1D action space
        sigma = np.std(top_particles, axis=0)

    # Sample new_particles
    if prob_vars.action_dim > 1:
        new_particles = np.random.multivariate_normal(mu.flatten(), sigma, size=num_particles)
    else:
        new_particles = np.random.normal(mu, sigma, size=(num_particles, horizon))

    if prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "LunarLanderContinuous" or prob_vars.prob == "PandaReacherDense":
        # print("best_cost ", best_cost, "\n")
        # print("best_action_sequence ", best_action_sequence, "\n")
        
        particles[prob_vars.num_particles-prob_vars.nb_random:] = np.random.uniform(prob_vars.action_low, prob_vars.action_high, (prob_vars.nb_random, prob_vars.horizon*prob_vars.action_dim))
        best_first_action = best_action_sequence[:prob_vars.action_dim] # .item()
    else: # Pendulum, MountainCarContinuous
        particles[prob_vars.num_particles-prob_vars.nb_random:] = np.random.uniform(prob_vars.action_low, prob_vars.action_high, (prob_vars.nb_random, prob_vars.horizon))
        best_first_action = best_action_sequence[0].item()

    return best_first_action, new_particles


"""
def discrete_cem_func(prob_vars, particles, costs, best_action_sequence):

    # print("costs.shape ", costs.shape, "\n")
    costs = costs.sum(dim=1)
    # print("costs.shape ", costs.shape, "\n")

    # num_sequences, sequence_length = particles.shape
    num_particles = prob_vars.num_particles
    horizon = prob_vars.horizon
    num_actions = prob_vars.nb_actions

    laplace_alpha = prob_vars.laplace_alpha

    # Get the indices of the top 15 particles
    print("prob_vars.nb_top_particles ", prob_vars.nb_top_particles, "\n")
    top_indices = torch.argsort(costs)[:prob_vars.nb_top_particles]
    top_particles = particles[top_indices]
    print("top_indices ", top_indices, "\n")

    position_probs = np.zeros((horizon, num_actions))
    print("top_particles.shape ", top_particles.shape, "\n")
    print("top_particles ", top_particles, "\n")

    for t in range(horizon):
        counts = np.bincount(top_particles[:, t].flatten(), minlength=num_actions)
        print("top_particles[:, t].shape ", top_particles[:, t].shape, "\n")
        print("counts ", counts, "\n")
        smoothed_counts = counts + laplace_alpha
        position_probs[t] = smoothed_counts / smoothed_counts.sum()

    new_particles = np.zeros((prob_vars.num_particles, horizon), dtype=int)
    
    for t in range(horizon):
        new_particles[:, t] = np.random.choice(num_actions, size=num_particles, p=position_probs[t])

    return new_particles
"""

"""
def continuous_cem_func(prob_vars, particles, costs, best_action_sequence, noisy=False):

    print("costs.shape ", costs.shape, "\n")
    costs = costs.sum(dim=1)
    print("costs.shape ", costs.shape, "\n")

    alpha = 0.7 # Update rate
    
    if noisy:
        a = 5
        b = 100

    # num_sequences, sequence_length = particles.shape
    num_particles = prob_vars.num_particles
    horizon = prob_vars.horizon
    num_actions = prob_vars.nb_actions

    # mu0 = np.mean(particles, axis=0)

    # if prob_vars.action_dim > 1:
    #     sigma0 = np.std(particles)
    # else:
    #     sigma0 = np.cov(particles, rowvar=False, bias = True)

    
    # Get the indices of the top 15 particles
    top_indices = torch.argsort(costs)[:prob_vars.nb_top_particles]
    top_particles = particles[top_indices]

    mu = np.mean(top_particles, axis=0)
    # new_mu = np.mean(top_particles, axis=0)
    # sigma = np.std(top_particles, axis=0)

    # Smoothing
    # mu = alpha * new_mu + (1 - alpha) * mu

    if prob_vars.action_dim > 1:
        sigma = np.cov(top_particles, rowvar=False, bias = True)
        # new_sigma = np.cov(top_particles, rowvar=False, bias = True)

        # # Smoothing and add noise if so
        # if noisy:
        #     noise = max(0, a-num_particles/b)
        #     noise = np.diag([noise]*horizon)
        #     sigma = alpha**2 * new_sigma + (1 - alpha)**2 * (sigma+noise)
        # else:
        #     sigma = alpha**2 * new_sigma + (1 - alpha)**2 * sigma

    else: # 1D action space
        print("In here \n")
        sigma = np.std(top_particles, axis=0)
        # new_sigma = np.std(top_particles, axis=0)

        # # # Smoothing and add noise if so
        # if noisy:
        #     noise = max(0, a-num_particles/b)
        #     sigma = alpha**2 * new_sigma + (1 - alpha)**2 * (sigma+noise)
        # else:
        #     sigma = alpha**2 * new_sigma + (1 - alpha)**2 * sigma
    
    print("mu.shape ", mu.shape, "\n")
    print("sigma.shape ", sigma.shape, "\n")

    # Sample new_particles
    if prob_vars.action_dim > 1:
        # new_particles = np.random.multivariate_normal(mu, sigma, size=num_particles)
        new_particles = np.random.multivariate_normal(mu.flatten(), sigma, size=num_particles)
        # new_particles = np.random.multivariate_normal(mu.flatten(), sigma, size=(num_particles, horizon))

    else:
        new_particles = np.random.normal(mu, sigma, size=(num_particles, horizon))

    return new_particles

"""


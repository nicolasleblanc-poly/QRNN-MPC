import numpy as np

def generate_random_action_sequences(prob_vars):
    # Generate new action sequences by sampling uniformly

    if prob_vars.prob == "CartPole":
        particles = np.random.randint(0, 2, (prob_vars.num_particles, prob_vars.horizon))
    elif prob_vars.prob == "Acrobot" or prob_vars.prob == "MountainCar": 
        particles = np.random.randint(0, 3, (prob_vars.num_particles, prob_vars.horizon))
    elif prob_vars.prob == "LunarLander":
        particles = np.random.randint(0, 4, (prob_vars.num_particles, prob_vars.horizon))
    elif prob_vars.prob == "PandaReacher" or prob_vars.prob == "MuJoCoReacher" or prob_vars.prob == "PandaPusher" or prob_vars.prob == "MuJoCoPusher" or prob_vars.prob == "LunarLanderContinuous" or prob_vars.prob == "PandaReacherDense" or prob_vars.prob == "PandaPusherDense":
        particles = np.random.uniform(prob_vars.action_low, prob_vars.action_high, (prob_vars.num_particles, prob_vars.action_dim*prob_vars.horizon))
    else: # Pendulum, MountainCarContinuous
        particles = np.random.uniform(prob_vars.action_low, prob_vars.action_high, (prob_vars.num_particles, prob_vars.horizon))

    return particles

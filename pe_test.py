import gymnasium as gym

env = gym.make("Pendulum-v1")
env.reset()
print(env.unwrapped.state)
# env.state


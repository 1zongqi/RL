import gym_examples
import gym
import time

env = gym.make('gym_examples/SortingCenter-v0', render_mode="human", disable_env_checker=True)
obs, info = env.reset(seed=42)
for _ in range(1000):
    actions = [env.action_space.sample() for _ in range(env.config['num_agents'])]
    obs, reward, terminated, truncated, info = env.step(actions)
    if terminated or truncated:
        obs, info = env.reset()
env.close()

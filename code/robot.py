import gym
env = gym.make('FetchReach-v1')
env.reset()
for _ in range(1):
    env.render()
    print(env.observation_space.spaces['observation'])
    print(env.spec.max_episode_steps)
    env.step(env.action_space.sample()) # take a random action

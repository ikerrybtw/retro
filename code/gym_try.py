import gym
env = gym.make('BipedalWalkerHardcore-v2')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        # env.render()
        print(observation)
        action = [0, 0, 0, 0]
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

import gymnasium as gym

env = gym.make('ALE/Seaquest-v5')
obs, info = env.reset()

for _ in range(10):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    obs, reward, terminated, truncated, info = env.step(action)

    print(f'Observation shape {obs.shape}')
    print(f'Reward {reward}')
    print(f'Terminated {terminated}')
    print(f'Truncated {truncated}')
    print(f'Info {info}')

    if terminated or truncated:
        obs, info = env.reset()

env.close()
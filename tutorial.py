import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

environment_name = 'CartPole-v1'

log_path = os.path.join('Training', 'Logs')
env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=20000)

PPO_path = os.path.join('Training', 'Models_Saved','PPO_Cartpole')
model.save(PPO_path)

del model

env = gym.make(environment_name, render_mode="human")

model = PPO.load(PPO_path, env=env)

evaluate_policy(model, env, n_eval_episodes=4, render=True)
env.close()

env = gym.make(environment_name, render_mode="human")

episodes = 5
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, tmp, info = env.step(action)
        score += reward
        
    print('Episode:{} Score:{}'.format(episode, score))
env.close()
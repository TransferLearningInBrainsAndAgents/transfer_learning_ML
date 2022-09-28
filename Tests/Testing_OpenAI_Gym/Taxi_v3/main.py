
from Tests.Testing_OpenAI_Gym.Taxi_v3.agent import Agent
from Tests.Testing_OpenAI_Gym.Taxi_v3.monitor import interact
import gym

env = gym.make('Taxi-v3', new_step_api=True)
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)
print(avg_rewards, best_avg_reward)
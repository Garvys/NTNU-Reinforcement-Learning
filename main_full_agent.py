from policy import *
from replay_memory import *
from full_agent import *
from live_plot import *
#from bootstrapped_dqn import *
import gym
import numpy as np
import time
import datetime

# Environment settings
max_episodes = 300
max_total_reward = 500
num_episodes_explorating = 1000

# Agent settings
network_layers = [400]

# Initialize simulation
envName = 'CartPole-v1'

env = gym.make(envName)

outdir = '/tmp/RL-experiment'

#Monitor
env.monitor.start(outdir, force=True, seed=0)
plotter = LivePlot(max_episodes, max_total_reward)

# Create agent

q_learning_config = QLearningConfig(batch_size = 200, gamma = 0.99, size_replay_min_to_train = 5, 
    learning_rate_start = 2e-3, learning_rate_end = 1e-7, time_learning_rate_end = 50000)
policy = UCB1Policy(env.action_space.n)

replay_memory_config = ReplayMemoryConfig(memorySize = 500000, use_prioritized_replay = True, alpha = 0.7,
        beta_zero = 0.7, total_steps = 40000)

dueling_config = DuelingConfig(use_dueling = False, size_net_value = 200, size_net_adv = 200)
bootstrapping_config = BootstrappingConfig(nb_heads = 1)

agent = DoubleDQNAgent(env.action_space, env.observation_space, main_network_layers = network_layers,
    q_learning_config = q_learning_config, policy = policy, replay_memory_config = replay_memory_config,
    bootstrapping_config = bootstrapping_config, dueling_config = dueling_config)

# Run simulation
rewards = []

for episode in range(max_episodes):

    #Reset the environment
    observation, reward, done = env.reset(), 0.0, False

    #Sum of the reward during the episode
    total_reward = 0.0

    #Episode not finished
    done = False

    agent.new_episode()

    while not done:
        
        #Policy select action
        action = agent.act(observation, reward, done)
        old_state = observation

        #Execute action in simulator
        observation, reward, done, _ = env.step(action)
        new_state = observation

        agent.update_network(old_state, action, reward, new_state, done)

        #Add to the rewards got during the episode
        total_reward += reward

    s = ''
    my_mean = -1
    if episode > 100:
        my_mean = sum(rewards[len(rewards) - 100:])/100.0
        s = 'Mean : {}'.format(my_mean)

    print("{} - {} {}".format(episode, total_reward, s))
    print('Timesteps : {}'.format(agent._time))
    rewards.append(total_reward)
    plotter.plot(rewards)

env.monitor.close()
file.close()

#gym.upload(outdir, api_key='sk_3qdNeOXT7ujek0tSWxBnQ')

#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
from math import sqrt

def get_returns(env, Q_func, epsilon, number_of_episodes=1,max_steps = 1000):
    
    returns = []
    
    state, info = env.reset()
    done, truncated = False, False
    t=0
    #print(f"\n=== Episode {ep} ===")
    while not (done or truncated) and t < max_steps:
        # choose a random action for now
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_func[state])
        next_state, reward, done, truncated, info = env.step(action)
        returns.append((state, action, reward, next_state))

        #print(f"Step {t:3d}: state={obs}, action={action}, "
        #      f"reward={reward}, next_state={next_obs}, done={done}")
        t += 1
        state = next_state

    return returns

def get_episode_return_stats(returns):
    list_of_rewards = []
    for state, action, reward, next_state in returns:
        list_of_rewards.append(reward)
    return (min(list_of_rewards),max(list_of_rewards),sum(list_of_rewards))

#Monte Carlo Control
def MC_control(gamma=0.99, max_episode_training=100000):
    env = gym.make("Taxi-v3")
    episode_return_stats = []
    Q_func = np.zeros((env.observation_space.n, env.action_space.n))
    n = np.zeros((env.observation_space.n, env.action_space.n))

    current_episode = 1
    while current_episode < max_episode_training:
        epsilon = 1 / max(1, current_episode - 10**4)
        returns = get_returns(env, Q_func, epsilon)
        episode_return_stats.append(get_episode_return_stats(returns))
        G = 0
        visited = set()
        for step in reversed(range(len(returns))):
            state, action, reward, next_state = returns[step]
            G = gamma*G + reward
            if (state, action) not in visited:
                visited.add((state,action))
                n[state][action] += 1
                Q_func[state][action] = Q_func[state][action] + 1/(n[state][action]) * (G - Q_func[state][action])
            
        if current_episode % 500 == 0:
            print(f"Episode {current_episode}, total reward: {sum([r for (_,_,r,_) in returns])}")

        current_episode += 1
    env.close()
    return Q_func, episode_return_stats

def policy_random(max_episode_training=50000, max_steps=50):
    env = gym.make("Taxi-v3")
    total_returns = []
    for episode in range(1, max_episode_training):
        state, info = env.reset()
        done, truncated = False, False
        returns = []
        t=0
        #print(f"\n=== Episode {ep} ===")
        while not (done or truncated) and t < max_steps:
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            returns.append((state, action, reward, next_state))

            #print(f"Step {t:3d}: state={obs}, action={action}, "
            #      f"reward={reward}, next_state={next_obs}, done={done}")
            t += 1
            state = next_state
        total_returns.append(get_episode_return_stats(returns))
    env.close()
    return total_returns

def plot_training_progress(episode_return_stats, group_size=100, title="Training Progress"):

    mins, maxs, totals = zip(*episode_return_stats)
    mins, maxs, totals = np.array(mins), np.array(maxs), np.array(totals)

    num_episodes = len(totals)
    num_groups = num_episodes // group_size

    avg_rewards, min_rewards, max_rewards = [], [], []

    for i in range(num_groups):
        start = i * group_size
        end = start + group_size
        batch = totals[start:end]
        avg_rewards.append(np.mean(batch))
        min_rewards.append(np.min(batch))
        max_rewards.append(np.max(batch))

    group_indices = np.arange(1, num_groups + 1) * group_size

    plt.figure(figsize=(10, 6))
    plt.plot(group_indices, avg_rewards, label="avg", color="blue")
    plt.plot(group_indices, min_rewards, label="min", color="orange")
    plt.plot(group_indices, max_rewards, label="max", color="green")
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def get_action(env,Q_func, state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q_func[state]) 

def sarsa(gamma=0.99, alpha=.1, max_episode_training=10000, max_step=1000):
    #initializations
    env = gym.make("Taxi-v3")
    Q_func = np.zeros((env.observation_space.n,env.action_space.n))
    current_episode = 1
    return_stats = []

    #each episode iteration
    while current_episode < max_episode_training:
        #more initializations
        state, info = env.reset()
        done, truncated = False, False
        t=0
        epsilon = 1 / max(1, current_episode - 10**4)
        action = get_action(env,Q_func, state, epsilon)
        episode_return = []
        
        # finding all S,A,R,S',A' in the current episode
        while not (done or truncated) and t < max_step:

            next_state, reward, done, truncated, info = env.step(action)
            next_action = get_action(env, Q_func, next_state, epsilon)
            Q_func[state][action] = Q_func[state][action] + alpha * (reward + gamma*Q_func[next_state][next_action] - Q_func[state][action])
            
            episode_return.append((state,action,reward,next_state))
            state = next_state
            action = next_action
            t += 1
        current_episode += 1
        return_stats.append(get_episode_return_stats(episode_return))

    return Q_func, return_stats

def Q_learning(gamma=0.99, alpha=.1, max_episode_training=10000, max_step=1000):
    
    #initializations
    env = gym.make("Taxi-v3")
    Q_func = np.zeros((env.observation_space.n,env.action_space.n))
    current_episode = 1
    return_stats = []


    #each episode iteration
    while current_episode < max_episode_training:
        #more initializations
        state, info = env.reset()
        done, truncated = False, False
        t=0
        epsilon = 1 / max(1, current_episode - 10**4)
        episode_returns = []
        
        # finding all S,A,R,S',A' in the current episode
        while not (done or truncated) and t < max_step:
            
            action = get_action(env, Q_func, state,epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            
            #this gives us the greedy Q-value for the next state
            next_action = np.argmax(Q_func[next_state])
            
            Q_func[state][action] = Q_func[state][action] + alpha * (reward + gamma*Q_func[next_state][next_action] - Q_func[state][action])
            
            episode_returns.append((state,action,reward,next_state))
            state = next_state
            t +=1
        current_episode += 1
        return_stats.append(get_episode_return_stats(episode_returns))
    
        if current_episode % 500 == 0:
            avg_return = np.mean([r[2] for r in episode_returns]) if episode_returns else 0
            print(f"Episode {current_episode}, total reward: {sum([r[2] for r in episode_returns])}")


    return Q_func, return_stats

if __name__ == "__main__":
    
    Monte_Q_func, Monte_episode_return_stats = MC_control(max_episode_training=30000)

    sarsa_Q_func, sarsa_episode_return_stats = sarsa(max_episode_training=30000)

    Q_learning_Q_func, Q_learning_episode_return_stats = Q_learning(max_episode_training=30000)

    plot_training_progress(Monte_episode_return_stats, title="Monte Carlo Training Progress")

    plot_training_progress(sarsa_episode_return_stats, title="SARSA Training Progress")

    plot_training_progress(Q_learning_episode_return_stats, title="Q_learning Training Progress")
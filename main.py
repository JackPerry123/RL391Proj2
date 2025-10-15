#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd

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
def MC_control(gamma=0.9, max_episode_training=100000):
    env = gym.make("Taxi-v3")
    episode_return_stats = []
    Q_func = np.zeros((env.observation_space.n, env.action_space.n))
    n = np.zeros((env.observation_space.n, env.action_space.n))

    number_of_episodes = 1
    while number_of_episodes < max_episode_training:
        epsilon = max(0.05, 1/number_of_episodes**.3)
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
            
        if number_of_episodes % 500 == 0:
            print(f"Episode {number_of_episodes}, total reward: {sum([r for (_,_,r,_) in returns])}")

        number_of_episodes += 1
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

def plot_training_progress(episode_return_stats, group_size=100):

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
    plt.title("Taxi Monte Carlo Training Progress")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def sarsa(gamam=0.9, max_episode_training=10000, max_step=1000):
    #initializations
    env = gym.make("Taxi-v3")
    Q_func = np.zeros((env.observation_space.n,env.action_space.n))
    current_episode = 0
    
    #each episode iteration
    while current_episode < max_episode_training:
        #more initializations
        state, info = env.reset()
        done, truncated = False, False
        t=0
        epsilon = 1/ max(0.05, 1/number_of_episodes**.3)
        alpha = max(.05, 1/sqrt(current_episode))
        
        #updating the current policy
        policy = np.zeros((env.observation_space.n,env.action_space.n))
       
        for state in range(env.observation_space):
            if np.random.rand() < epsilon:
                policy[state] = env.action_space.sample()
            else:
                policy[state] = np.argmax(Q_func[state])
        
        # finding all S,A,R,S',A' in the current episode
        while not (done or truncated) and t < max_step:
            action = policy[state]
            next_state, reward, done, truncated, info = env.step(action)
            next_action = policy[next_state]
            Q_func = Q_func + alpha * (reward + gamma*Q_func[next_state][next_action] - Q_func[state][action])
            state = next_state
        
def Q_learning(gamam=0.9, max_episode_training=10000, max_step=1000):

    #initializations
    env = gym.make("Taxi-v3")
    Q_func = np.zeros((env.observation_space.n,env.action_space.n))
    current_episode = 0
    
    #each episode iteration
    while current_episode < max_episode_training:
        #more initializations
        state, info = env.reset()
        done, truncated = False, False
        t=0
        epsilon = 1/ max(0.05, 1/number_of_episodes**.3)
        alpha = max(.05, 1/sqrt(current_episode))
        
        #updating the current policy
       
       
        for state in range(env.observation_space):
            if np.random.rand() < epsilon:
                policy[state] = env.action_space.sample()
            else:
                policy[state] = np.argmax(Q_func[state])
        
        # finding all S,A,R,S',A' in the current episode
        while not (done or truncated) and t < max_step:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_func[state])
            
            next_state, reward, done, truncated, info = env.step(action)
            
            #this gives us the greedy Q-value for the next state
            next_action = np.argmax(Q_func[next_state])
            
            Q_func = Q_func + alpha * (reward + gamma*Q_func[next_state][next_action] - Q_func[state][action])
            state = next_state
        




if __name__ == "__main__":
    
    Q_func, episode_return_stats = MC_control(max_episode_training=100000)
    #print("Q_func", Q_func)
    #print("episode_return_stats",episode_return_stats)

    #episode_return_stats2 = policy_random(max_episode_training=10000)

    plot_training_progress(episode_return_stats)

    #plot_no_window_stats(episode_return_stats2)

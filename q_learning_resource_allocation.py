"""
Q-Learning for Dynamic Resource Allocation in 5G URLLC Networks
================================================================
MSc Dissertation - Loughborough University (2021) - Distinction
Author: Puja Chattopadhyay

Description:
    Implements a Q-Learning algorithm to dynamically allocate channels
    to users in a 5G Ultra Reliable Low Latency Communications (URLLC)
    network. The problem is modelled as a finite Markov Decision Process
    (MDP) using Instantaneous Channel State Information (ICSI) and
    packet queue lengths as state inputs.

Dependencies:
    numpy, matplotlib, scipy
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sct
from itertools import product


# ─────────────────────────────────────────
# NETWORK CONFIGURATION
# ─────────────────────────────────────────

total_users = 3
total_channels = 2
max_packet_in_queue = 3   # max possible packets in the queue per user
queue_limit = 3
total_states = pow(total_users, queue_limit)


# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def multichoose(n, m):
    """Generate all valid action matrices (channel-to-user assignments)."""
    x1 = product([0, 1], repeat=n * m)
    x1 = np.reshape(list(x1), (-1, n, m))
    x = []
    for i in range(len(x1)):
        if (np.sum(x1[i]) == 2 or np.sum(x1[i]) == 0) and not np.any(x1[i].sum(axis=0) > 1):
            x.append([x1[i]])
    return x


def random_combinations(length, queue_range):
    """Generate all possible state combinations based on queue lengths."""
    comb_set = [[list(range(0, queue_range))] for _ in range(length)]
    h = list(np.array(np.meshgrid(*comb_set)).T.reshape(-1, len(comb_set)))
    return h


def truncated_poisson(mu, max_value, size):
    """Sample from a truncated Poisson distribution."""
    temp_size = size
    while True:
        temp_size *= 2
        temp = sct.poisson.rvs(mu, size=temp_size)
        truncated = temp[temp <= max_value]
        if len(truncated) >= size:
            return truncated[np.random.randint(0, len(truncated) - 1)]


def find_row_index(states_list, current_state):
    """Return the row index of the current state in the states table."""
    for j in range(total_states):
        if list(states_list[j]) == current_state:
            return j


def randbin(M, N, P):
    """Generate a binary channel gain matrix using Bernoulli distribution."""
    return np.random.choice([1, 0], size=(M, N), p=[P, 1 - P])


def choose_action(actions, col, rand_col, epsilon):
    """
    Choose an action based on epsilon-greedy policy.
    - If random number > epsilon: explore (random action)
    - Else: exploit (best known action)
    """
    ec = np.random.rand()
    if ec > epsilon:
        g = np.array(range(0, rand_col))
        g1 = np.delete(g, col)
        chosen_act = actions[np.random.choice(g1)]
    else:
        chosen_act = actions[col]
    return chosen_act


def calc_reward(packets, gain, action, num_channel):
    """Calculate reward, packet drop, and packet income for each user."""
    reward = np.zeros((3, 1))
    drop = np.zeros((3, 1))
    income = np.zeros((3, 1))
    mat = np.sum(np.multiply(action, gain), axis=-1)

    for i in range(len(packets)):
        action_gain = 0
        if packets[i] != 0:
            action_gain += mat[0][i]
            reward[i] = (min(action_gain, packets[i]) / packets[i]) * 100
            income[i] = packets[i]
            drop[i] = max(packets[i] - action_gain, 0)
        else:
            reward[i] = 0

    return reward, drop, income


def get_action_index(actions, a):
    """Return the index of action 'a' in the actions list."""
    for k in range(len(actions)):
        if np.array_equal(actions[k], a, equal_nan=False):
            return k


def avg_per_interval(values, num, interval):
    """Calculate average reward per interval of episodes for plotting."""
    value_per_interval = np.split(np.array(values), (num - 2) / interval)
    count = interval
    avg_rewards = []
    counts = []
    for r in value_per_interval:
        avg_rewards.append(sum(r) / interval)
        counts.append(count)
        count += interval
    return counts, avg_rewards


# ─────────────────────────────────────────
# STATE & ACTION SPACE INITIALISATION
# ─────────────────────────────────────────

states = random_combinations(total_users, max_packet_in_queue)
actions = multichoose(3, 2)
total_actions = len(actions)


# ─────────────────────────────────────────
# Q-LEARNING HYPERPARAMETERS
# ─────────────────────────────────────────

q_table = np.zeros((total_states, total_actions))
epsilon = 0.01          # initial exploration threshold
discount_factor = 0.987  # gamma: weight for future rewards
learning_rate = 0.09    # alpha: how fast the agent learns
number_episodes = 60002

# Truncated Poisson parameters for packet arrival
mu = [0.5, 0.5, 0.5]
max_value = 2


# ─────────────────────────────────────────
# TRACKING VARIABLES
# ─────────────────────────────────────────

ep_q_value, ep_rewards = [], []
ep_rew1, ep_rew2, ep_rew3, ep = [], [], [], []
p_in1, p_in2, p_in3 = [], [], []
p_drop1, p_drop2, p_drop3 = [], [], []
cumulative_reward = 0


# ─────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────

for t in range(2, number_episodes):

    bernoulli_prob = 0.3
    min_exp = 0.99

    # Sample packet arrivals for each user (current state)
    packets_arrived = [truncated_poisson(mu[k], max_value, 3) for k in range(total_users)]

    # Find current state index in state table
    row_index = find_row_index(states, packets_arrived)

    # Select action with highest Q-value for current state
    col_index = np.where(q_table[row_index][:] == np.max(q_table[row_index]))
    if len(col_index[0][:]) > 1:
        col_index = np.random.choice(col_index[0][:])
    else:
        col_index = col_index[0][0]

    # Apply epsilon-greedy policy to choose final action
    final_action = choose_action(actions, col_index, total_actions, epsilon)
    col_index = get_action_index(actions, final_action)

    # Sample channel gain matrix
    channel_matrix = randbin(total_users, total_channels, bernoulli_prob)

    # Calculate reward for this (state, action) pair
    reward_matrix, drop, income = calc_reward(packets_arrived, channel_matrix, final_action, total_channels)

    # Q-value update using Bellman equation
    old_q_value = q_table[row_index][col_index]
    new_q = np.delete(q_table, row_index, axis=0)
    new_q = np.delete(new_q, col_index, axis=1)
    future_q_value = new_q[random.randint(0, total_states - 2)][random.randint(0, total_actions - 2)]

    temporal_difference = np.sum(reward_matrix) + (discount_factor * future_q_value) - old_q_value
    new_q_value = old_q_value + (learning_rate * temporal_difference)
    q_table[row_index, col_index] = new_q_value

    # Decay epsilon (shift from exploration toward exploitation over time)
    epsilon = min(epsilon + (0.0001 * math.log(t)), min_exp)

    # ── Record metrics ──
    ep_q_value.append(new_q_value)
    ep_rewards.append(np.sum(reward_matrix))
    ep.append(t)
    ep_rew1.append(reward_matrix[0][0])
    ep_rew2.append(reward_matrix[1][0])
    ep_rew3.append(reward_matrix[2][0])
    p_in1.append(income[0][0])
    p_in2.append(income[1][0])
    p_in3.append(income[2][0])
    p_drop1.append(drop[0][0])
    p_drop2.append(drop[1][0])
    p_drop3.append(drop[2][0])


# ─────────────────────────────────────────
# RESULTS & PLOTTING
# ─────────────────────────────────────────

div = 1000

x_val2, y_val2 = avg_per_interval(ep_rewards, number_episodes, div)
x_val3, y_val3 = avg_per_interval(ep_rew1, number_episodes, div)
x_val4, y_val4 = avg_per_interval(ep_rew2, number_episodes, div)
x_val5, y_val5 = avg_per_interval(ep_rew3, number_episodes, div)
x_in1, y_in1 = avg_per_interval(p_in1, number_episodes, div)
x_in2, y_in2 = avg_per_interval(p_in2, number_episodes, div)
x_in3, y_in3 = avg_per_interval(p_in3, number_episodes, div)
x_drop1, y_drop1 = avg_per_interval(p_drop1, number_episodes, div)
x_drop2, y_drop2 = avg_per_interval(p_drop2, number_episodes, div)
x_drop3, y_drop3 = avg_per_interval(p_drop3, number_episodes, div)

# Plot 1: Total average reward
plt.figure(1)
plt.plot(x_val2, y_val2)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title(f"Average total reward per {div} episodes")
plt.grid()
plt.show()

# Plot 2: Per-user cumulative reward
plt.figure(2)
plt.plot(x_val3, y_val3, label=f"User 1 | Bernoulli p=0.3 | mu={mu[0]}")
plt.plot(x_val4, y_val4, label=f"User 2 | Bernoulli p=0.5 | mu={mu[1]}")
plt.plot(x_val5, y_val5, label=f"User 3 | Bernoulli p=0.8 | mu={mu[2]}")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title(f"Average cumulative reward per user per {div} episodes")
plt.legend()
plt.grid()
plt.show()

# Plot 3: Packet income vs drop per user
plt.figure(3)
plt.plot(x_in1, y_in1, label="Packet income - User 1")
plt.plot(x_in2, y_in2, label="Packet income - User 2")
plt.plot(x_in3, y_in3, label="Packet income - User 3")
plt.plot(x_drop1, y_drop1, label="Packet drop - User 1")
plt.plot(x_drop2, y_drop2, label="Packet drop - User 2")
plt.plot(x_drop3, y_drop3, label="Packet drop - User 3")
plt.xlabel("Episodes")
plt.ylabel("Average number of packets")
plt.title(f"Packet income vs drop per user per {div} episodes")
plt.legend()
plt.grid()
plt.show()

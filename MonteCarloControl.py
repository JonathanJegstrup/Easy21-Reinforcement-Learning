from random import choice, random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from Easy21 import Action, Reward, Rules, State, step


def run_episode(N_dealer_states: int, N0: int, N: np.ndarray) -> Tuple[Reward, List[Tuple[int]]]:
    card_value_to_index = {i + 1: i for i in range(N_dealer_states)}
    player_sum_to_index = {
        i + Rules.MIN_VALUE.value: i for i in range(Rules.MAX_VALUE.value - Rules.MIN_VALUE.value + 1)
    }

    visited_indices = []
    state = State()

    terminate = False
    while terminate is False:
        # Update variables
        state_index = (card_value_to_index[state.dealers_first_card], player_sum_to_index[state.players_sum])
        eps = N0 / (N0 + N[state_index])
        N[state_index] += 1

        # Take action using eps-Greedy Exploration
        if 1 - eps >= random():
            max_action_value = Q[state_index[0], state_index[1]].max()
            action_ind = choice(np.where(Q[state_index[0], state_index[1]] == max_action_value)[0])
        else:
            action_ind = choice(range(N_actions))

        action = Action(action_ind)

        # Save relevant stuff
        state_action_index = (state_index[0], state_index[1], action_ind)
        visited_indices.append(state_action_index)

        # Take next step
        terminate, reward = step(state, action)

    return reward, visited_indices, N


def update_Q(Q: np.ndarray, G: int, visited_indices: List[Tuple[int]], N_a: np.ndarray) -> np.ndarray:
    for index in visited_indices:
        N_a[index] += 1
        alpha = 1 / N_a[index]
        Q[index] = Q[index] + alpha * (G - Q[index])

    return Q, N_a


def plot_V(V: np.ndarray, title: str = None) -> plt.figure:
    dealer_showing = np.arange(1, 11, 1)
    player_sum = np.arange(1, 22, 1)

    X, Y = np.meshgrid(dealer_showing, player_sum)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, np.swapaxes(V, 0, 1))
    ax.set_xlabel("Dealer showing")
    ax.set_ylabel("Player sum")
    ax.set_xticks(range(1, 11))
    ax.set_yticks(range(1, 22))
    ax.set_zticks([-1, 0, 1])
    ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Y) / 2))

    if title:
        plt.title(title)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    N_dealer_states = 10  # 10 dealer first card options
    N_player_states = 21  # 21 player sum options
    N_actions = 2  # Hit or Stick

    N0 = 100  # Constant
    N_episodes = 10_000_000

    # Init
    Q = np.zeros((N_dealer_states, N_player_states, N_actions), dtype="float")
    N = np.zeros((N_dealer_states, N_player_states), dtype="int")
    N_a = np.zeros((N_dealer_states, N_player_states, N_actions), dtype="int")

    # Run MC
    for episode in tqdm(range(N_episodes)):
        # Run episode
        reward, visited_indices, N = run_episode(N_dealer_states, N0, N)

        # Update information
        Q, N_a = update_Q(Q, reward.value, visited_indices, N_a)

    V = np.max(Q, axis=-1)
    fig = plot_V(V, title=f"MC with N_episodes = {N_episodes:,}")

    fig.savefig("optimal_V.png")

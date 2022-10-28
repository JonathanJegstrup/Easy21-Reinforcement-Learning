from random import choice, random
from typing import Dict, List, Tuple

import numpy as np
from easy21 import Action, Reward, Rules, State
from plotting import plot_V
from tqdm import tqdm


def run_episode(
    Q: np.ndarray, N0: int, N: np.ndarray, card_value_to_index: Dict[int, int], player_sum_to_index: Dict[int, int]
) -> Tuple[Reward, List[Tuple[int]]]:
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
        terminate, reward = state.step(action)

    return reward, visited_indices, N


def update_Q(Q: np.ndarray, G: int, visited_indices: List[Tuple[int]], N_a: np.ndarray) -> np.ndarray:
    for index in visited_indices:
        N_a[index] += 1
        alpha = 1 / N_a[index]
        Q[index] = Q[index] + alpha * (G - Q[index])

    return Q, N_a


if __name__ == "__main__":
    N_dealer_states = 10  # 10 dealer first card options
    N_player_states = 21  # 21 player sum options
    N_actions = 2  # Hit or Stick

    N0 = 100  # Constant
    N_episodes = 1_000_000

    # Init
    Q = np.zeros((N_dealer_states, N_player_states, N_actions), dtype="float")
    N = np.zeros((N_dealer_states, N_player_states), dtype="int")
    N_a = np.zeros((N_dealer_states, N_player_states, N_actions), dtype="int")

    card_value_to_index = {i + 1: i for i in range(N_dealer_states)}
    player_sum_to_index = {
        i + Rules.MIN_VALUE.value: i for i in range(Rules.MAX_VALUE.value - Rules.MIN_VALUE.value + 1)
    }

    # Run MC
    for episode in tqdm(range(N_episodes)):
        # Run episode
        reward, visited_indices, N = run_episode(Q, N0, N, card_value_to_index, player_sum_to_index)

        # Update information
        Q, N_a = update_Q(Q, reward.value, visited_indices, N_a)

    V = np.max(Q, axis=-1)
    fig = plot_V(V, title=f"MC with N_episodes = {N_episodes:,}")

    fig.savefig("Easy21/Plots/test.png")

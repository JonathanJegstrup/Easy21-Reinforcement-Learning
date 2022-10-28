from random import choice, random
from typing import Tuple

import numpy as np
from easy21 import Action, Rules, State
from plotting import plot_V
from tqdm import tqdm


class TDControl:
    # Game specific constants
    N_dealer_states = 10  # 10 dealer first card options
    N_player_states = 21  # 21 player sum options
    N_actions = 2  # Hit or Stick

    def __init__(self, lam: float = 1.0, N0: int = 100) -> None:
        self.lam = lam
        self.N0 = N0

        # Initialize
        self.Q = np.zeros((self.N_dealer_states, self.N_player_states, self.N_actions), dtype="float")
        self.N = np.zeros((self.N_dealer_states, self.N_player_states), dtype="int")
        self.N_a = np.zeros((self.N_dealer_states, self.N_player_states, self.N_actions), dtype="int")
        self.N_episodes_run = 0

        self.card_value_to_index = {i + 1: i for i in range(self.N_dealer_states)}
        self.player_sum_to_index = {
            i + Rules.MIN_VALUE.value: i for i in range(Rules.MAX_VALUE.value - Rules.MIN_VALUE.value + 1)
        }

    def decide_action(self, eps: float, state_index: Tuple[int, int]) -> Tuple[Tuple[int, int, int], Action]:
        # Take action using eps-Greedy Exploration
        if 1 - eps >= random():
            max_action_value = self.Q[state_index[0], state_index[1]].max()
            action_ind = choice(np.where(self.Q[state_index[0], state_index[1]] == max_action_value)[0])
        else:
            action_ind = choice(range(self.N_actions))

        action = Action(action_ind)
        state_action_index = (state_index[0], state_index[1], action_ind)

        return state_action_index, action

    def run_episode(self) -> np.ndarray:
        # Initialize
        E = np.zeros_like(self.Q)  # Eligibility traces
        state = State()
        state_index = (self.card_value_to_index[state.dealers_first_card], self.player_sum_to_index[state.players_sum])

        # Take first action
        eps = self.N0 / (self.N0 + self.N[state_index])
        state_action_index, action = self.decide_action(eps, state_index)
        terminate, reward = state.step(action)

        # Update state counter
        self.N[state_index] += 1
        self.N_a[state_action_index] += 1

        # Run episode
        while terminate is False:
            # Decide another action from this new state
            new_state_index = (
                self.card_value_to_index[state.dealers_first_card],
                self.player_sum_to_index[state.players_sum],
            )
            eps = self.N0 / (self.N0 + self.N[new_state_index])
            new_state_action_index, new_action = self.decide_action(eps, new_state_index)

            # Update variables
            delta = reward.value + self.Q[new_state_action_index] - self.Q[state_action_index]
            alpha = 1 / self.N_a[state_action_index]
            E[state_action_index] += 1

            # Update Q and E
            self.Q = self.Q + alpha * delta * E
            E = self.lam * E

            # Set new -> old
            state_index = new_state_index
            state_action_index = new_state_action_index
            action = new_action

            # Take step
            terminate, reward = state.step(action)
            self.N[state_index] += 1
            self.N_a[state_action_index] += 1

        # Make final update to Q
        E[state_action_index] += 1
        alpha = 1 / self.N_a[state_action_index]
        self.Q = self.Q + alpha * (reward.value - self.Q[state_action_index]) * E

        self.N_episodes_run += 1

        return self.Q

    def run_N_episodes(self, N: int = 1000) -> np.ndarray:
        for _ in tqdm(range(N)):
            self.run_episode()

    def return_V(self) -> np.ndarray:
        return np.max(self.Q, axis=-1)


if __name__ == "__main__":
    LAMBDA = 1.0
    N0 = 100
    N_EPISODES = 1000000

    control = TDControl(LAMBDA, N0)
    control.run_N_episodes(N_EPISODES)
    V = control.return_V()

    fig = plot_V(V, title=f"TD with N_episodes = {control.N_episodes_run:,}, and lambda = {control.lam:.2f}")
    fig.savefig("Easy21/Plots/TD_optimal_V.png")

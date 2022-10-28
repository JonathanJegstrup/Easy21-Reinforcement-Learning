from random import choice, random
from typing import Tuple

import numpy as np
from logic.easy21 import Action, Rules


class BaseControl:
    # Game specific constants
    N_dealer_states = 10  # 10 dealer first card options
    N_player_states = 21  # 21 player sum options
    N_actions = 2  # Hit or Stick

    def __init__(self, N0: int = 100) -> None:
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

    def eps_greedy_action(self, eps: float, state_index: Tuple[int, int]) -> Tuple[Tuple[int, int, int], Action]:
        # Take action using eps-Greedy Exploration
        if 1 - eps >= random():
            max_action_value = self.Q[state_index[0], state_index[1]].max()
            action_ind = choice(np.where(self.Q[state_index[0], state_index[1]] == max_action_value)[0])
        else:
            action_ind = choice(range(self.N_actions))

        action = Action(action_ind)
        state_action_index = (state_index[0], state_index[1], action_ind)

        return state_action_index, action

    def return_V(self) -> np.ndarray:
        V = np.max(self.Q, axis=-1)
        return V

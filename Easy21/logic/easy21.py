from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from random import randint, random
from typing import Tuple


class Color(Enum):
    RED = auto()
    BLACK = auto()


class Action(Enum):
    HIT = 0
    STICK = 1


class Reward(Enum):
    LOSE = -1
    DRAW = 0
    WIN = 1
    BUST = -1
    NO_REWARD = 0


class Rules(Enum):
    MAX_VALUE = 21
    MIN_VALUE = 1
    DEALER_THRESHOLD = 17


@dataclass
class Card:
    value: int
    color: Color


class State:
    def __init__(self) -> None:
        self.players_sum = randint(1, 10)
        self.player_stick = False
        self.dealers_first_card = randint(1, 10)
        self.dealer_stick = False

        self._dealers_sum = deepcopy(self.dealers_first_card)

    def dealer_turn(self) -> bool:
        if self._dealers_sum < Rules.DEALER_THRESHOLD.value:
            terminal, self._dealers_sum = hit(self._dealers_sum)
        else:
            self.dealer_stick = True
            terminal = False

        return terminal

    def step(self, action: Action) -> Tuple[bool, Reward]:
        terminal = False
        # Agent action / Player turn
        if action == Action.HIT:
            terminal, self.players_sum = hit(self.players_sum)
            if terminal:
                return terminal, Reward.BUST

        elif action == Action.STICK:
            self.player_stick = True
        else:
            raise NotImplementedError("action not implemented")

        # Enviroment / Dealers turn
        if not self.dealer_stick:
            if self.player_stick:
                while self.dealer_stick is False and terminal is False:
                    terminal = self.dealer_turn()
            else:
                terminal = self.dealer_turn()

            if terminal:
                return terminal, Reward.WIN

        # Evaluate values if both players has stuck
        if self.player_stick & self.dealer_stick:
            terminal = True

            if self.players_sum > self._dealers_sum:
                return terminal, Reward.WIN
            elif self.players_sum < self._dealers_sum:
                return terminal, Reward.LOSE
            else:
                return terminal, Reward.DRAW

        return terminal, Reward.NO_REWARD


class Enviroment:
    def __init__(self, state: State) -> None:
        self.dealers_sum = state.dealers_first_card


def draw_card() -> Card:
    value = randint(1, 10)

    random_value = random()
    if random_value <= 1 / 3:
        color = Color.RED
    else:
        color = Color.BLACK

    card = Card(value, color)

    return card


def hit(sum: int) -> bool:
    card = draw_card()
    if card.color == Color.RED:
        new_sum = sum - card.value
    else:
        new_sum = sum + card.value

    terminal = new_sum > Rules.MAX_VALUE.value or new_sum < Rules.MIN_VALUE.value
    return terminal, new_sum


if __name__ == "__main__":
    state = State()
    terminal = False
    total_reward = 0
    while not terminal:
        print()
        print(f"Dealer's first card: {state.dealers_first_card}")
        print(f"Player's sum: {state.players_sum}")
        if not state.player_stick:
            action = input("HIT or STICK? ")
            action = getattr(Action, action)

        terminal, step_reward = state.step(action)
        total_reward = step_reward.value

    print()
    print(f"Player sum: {state.players_sum}, Dealers sum: {state._dealers_sum}")
    print(f"You {step_reward.name} with score {total_reward}")

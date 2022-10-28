from typing import List, Tuple

from base_control import BaseControl
from easy21 import Reward, State
from plotting import plot_V
from tqdm import tqdm


class MonteCarloControl(BaseControl):
    def run_episode(self) -> Tuple[Reward, List[Tuple[int]]]:
        visited_indices = []
        state = State()

        terminate = False
        while terminate is False:
            # Update variables
            state_index = (
                self.card_value_to_index[state.dealers_first_card],
                self.player_sum_to_index[state.players_sum],
            )
            eps = self.N0 / (self.N0 + self.N[state_index])
            self.N[state_index] += 1

            state_action_index, action = self.eps_greedy_action(eps, state_index)

            visited_indices.append(state_action_index)

            # Take next step
            terminate, reward = state.step(action)

        self.N_episodes_run += 1
        return reward, visited_indices

    def update_Q(self, reward: int, visited_indices: List[Tuple[int]]) -> None:
        for index in visited_indices:
            self.N_a[index] += 1
            alpha = 1 / self.N_a[index]
            self.Q[index] = self.Q[index] + alpha * (reward - self.Q[index])

    def run_N_update_steps(self, N: int = 1000) -> None:
        for _ in tqdm(range(N)):
            reward, visited_indices = self.run_episode()
            self.update_Q(reward.value, visited_indices)


if __name__ == "__main__":
    N0 = 100
    N_EPISODES = 1_000_000

    control = MonteCarloControl(N0)
    control.run_N_update_steps(N_EPISODES)
    V = control.return_V()

    fig = plot_V(V, title=f"MC with N_episodes = {control.N_episodes_run:,}")
    fig.savefig("Easy21/plots/MC_optimal_V.png")

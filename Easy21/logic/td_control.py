import numpy as np
from base_control import BaseControl
from easy21 import State
from plotting import plot_V
from tqdm import tqdm


class TDControl(BaseControl):
    def __init__(self, N0: int = 100, lam: float = 1.0) -> None:
        super().__init__(N0)
        self.lam = lam

    def run_episode(self) -> np.ndarray:
        # Initialize
        E = np.zeros_like(self.Q)  # Eligibility traces
        state = State()
        state_index = (self.card_value_to_index[state.dealers_first_card], self.player_sum_to_index[state.players_sum])

        # Take first action
        eps = self.N0 / (self.N0 + self.N[state_index])
        state_action_index, action = self.eps_greedy_action(eps, state_index)
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
            new_state_action_index, new_action = self.eps_greedy_action(eps, new_state_index)

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

    def run_N_episodes(self, N: int = 1000) -> None:
        for _ in tqdm(range(N)):
            self.run_episode()


if __name__ == "__main__":
    LAMBDA = 0.1
    N0 = 100
    N_EPISODES = 1_000_000

    control = TDControl(N0, LAMBDA)
    control.run_N_episodes(N_EPISODES)
    V = control.return_V()

    fig = plot_V(V, title=f"TD with N_episodes = {control.N_episodes_run:,}, and lambda = {control.lam:.2f}")
    fig.savefig("Easy21/plots/TD_optimal_V.png")

import matplotlib.pyplot as plt
import numpy as np
from logic.mc_control import MonteCarloControl
from logic.plotting import plot_V
from logic.td_control import TDControl

if __name__ == "__main__":
    # Constants
    N0 = 100
    N_mc_steps = 1_000_000
    N_td_steps = 10_000
    lambdas_to_record = [0.0, 1.0]

    # Run MC Control to get 'optimal' value-function
    print("Computing optimal Q with MC...")
    mc_control = MonteCarloControl(N0)
    mc_control.run_N_update_steps(N_mc_steps)
    mc_Q = mc_control.Q
    V = mc_control.return_V()

    # Save figure
    fig = plot_V(V, title=f"MC with N_episodes = {mc_control.N_episodes_run:,}")
    fig.savefig("plots/MC_optimal_V.png")

    # Run loop of TD control with different lambdas
    print("Looping through different lambdas...")
    mse = []
    learning_curves = []
    lambdas = np.arange(0, 1.1, 0.1)
    lambdas_to_record = sorted(lambdas_to_record)
    for lam in lambdas:
        td_control = TDControl(N0, lam)

        if lam in lambdas_to_record:
            learning_curve = [np.mean((mc_Q - td_control.Q) ** 2)]
            for _ in range(N_td_steps):
                td_control.run_episode()
                learning_curve.append(np.mean((mc_Q - td_control.Q) ** 2))

            learning_curves.append(learning_curve)

        else:
            td_control.run_N_episodes(N_td_steps)
        td_Q = td_control.Q

        # Compute mean squared error
        mse.append(np.mean((mc_Q - td_Q) ** 2))

    # MSE pr. lambda
    fig = plt.figure()
    plt.bar(lambdas, mse, 0.1, align="center", edgecolor="k")
    plt.xticks(lambdas)
    plt.xlabel("$\lambda$")
    plt.ylabel("MSE")
    plt.tight_layout()

    fig.savefig("plots/MSE_pr_lambda.png")

    # MSE pr. episode
    fig = plt.figure(figsize=(6.5 * len(learning_curves), 4.8))
    for i in range(len(learning_curves)):
        plt.subplot(1, len(learning_curves), i + 1)
        plt.title(f"$\lambda = {lambdas_to_record[i]}$")
        plt.plot(learning_curves[i])
        plt.xlabel("Episode")
        plt.ylabel("MSE")

    plt.tight_layout()

    fig.savefig("plots/MSE_pr_episode.png")

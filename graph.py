import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os


def plot_in_graph(dirs):
    values = ['EpExtrinsicReward/Average', 'intrinsic_rewards/Std', 'art_num_classes/Max']
    num_plots = len(values)
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 14))
    for dirname in dirs:
        name = dirname
        df = pd.read_csv(f'results/ppo_DeepmindOrdealEnv-v0/{dirname}/progress.csv')
        iterations = df['Diagnostics/Iteration']
        # df = [df['GameScore/Average'], df['intrinsic_rewards/Average']]
        # df = [df['EpAveExtrinsicReward/Average'], df['art_num_classes/Max']]
        for idx, (value, ax) in enumerate(zip(values, axes)):
            y = df[value].rolling(10).mean()
            plot_len = min(len(iterations), len(y))
            ax.plot(
                iterations[:plot_len],
                y[:plot_len], label=f'{name}'
            )
            ax.set_title(value)

    for ax in axes:
        ax.legend()

    axes[0].set_ylim((-1.1, 2.1))
    axes[1].set_yscale('log')
    plt.tight_layout()
    plt.show()
    # axes[0].set_yscale('log')
    # axes[1].set_ylim((0.00001, 0.01))

def main():
    sns.set_theme(style="darkgrid")
    # dirs = ('art', 'art_0', 'arthl_0', 'arthl_1', 'disagreement_0', 'icm_0', 'none_0')
    # dirs = [f'art_{i}' for i in (6, 7, 8)] + ['arthl_6', 'rnd_6']
    all_dirs = sorted(os.listdir("results/ppo_DeepmindOrdealEnv-v0/"))
    filters = [
        lambda d: 'art' in d,
        lambda d: 'rnd' in d,
        lambda d: 'icm' in d,
        lambda d: 'art' not in d and 'rnd' not in d and 'icm' not in d
    ]
    for f in filters:
        plot_in_graph(filter(f, all_dirs))


if __name__ == '__main__':
    main()

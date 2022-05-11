import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

def get_category(dirname):
    if 'arthl_7' in dirname or 'arthl_' in dirname:  # or 'arthl_' in dirname:
        return 'ART-HL'
    elif 'art_' in dirname:
        try:
            if int(dirname[-1]) > 5:
                return "ART"
        except ValueError:
            pass
    elif dirname in (f'icm_{i}' for i in (1, 2, 3)):
        return 'ICM'
    elif dirname in (f'rnd_{i}' for i in (1, 2, 3, 6, 7)):
        return 'RND'
    else:
        return None


    # return s.rolling(smoothing).mean()


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


def plot_matplotlib():
    sns.set_theme(style="darkgrid")
    all_dirs = sorted(os.listdir("results/ppo_DeepmindOrdealEnv-v0/"))
    filters = [
        lambda d: 'art' in d,
        lambda d: 'rnd' in d,
        lambda d: 'icm' in d,
        lambda d: 'art' not in d and 'rnd' not in d and 'icm' not in d
    ]
    for f in filters:
        plot_in_graph(filter(f, all_dirs))

def preprocess(dirnames, smoothing=100, max_its=400_000):
    dfs = {}
    for dirname in dirnames:
        agent_type = get_category(dirname)
        df = pd.read_csv(f'results/ppo_DeepmindOrdealEnv-v0/{dirname}/progress.csv')
        print(f"{agent_type}: {dirname}, len: {(len(df)*125)//1000}k iterations")
        if agent_type is None:
            continue
        # df = pd.read_csv(f'results/ppo_DeepmindOrdealEnv-v0/{dirname}/progress.csv')
        df = df.rolling(smoothing).mean()
        df :pd.DataFrame = df.assign(agent_type=agent_type)
        dfs[dirname] = df.where(df['Diagnostics/Iteration'] < max_its)

    longest_df = max(map(len, dfs.values()))

    full_df = pd.concat(list(dfs.values()), ignore_index=True)

    return full_df

def plot_sns(y_val):
    sns.set_theme(style="darkgrid")
    all_dirs = sorted(os.listdir("results/ppo_DeepmindOrdealEnv-v0/"))
    df = preprocess(all_dirs)
    sns.lineplot(
        data=df, x='Diagnostics/Iteration', y=y_val,
        ci='sd', hue='agent_type'
    )

# values = ['EpExtrinsicReward/Average', 'intrinsic_rewards/Std', 'art_num_classes/Max']
# iterations = df['Diagnostics/Iteration']

def plot_several():
    y_vals = ['EpExtrinsicReward/Average', ]
    # y_vals = ['intrinsic_rewards/Average', 'intrinsic_rewards/Std']
    for y_str in y_vals:
        plt.figure(figsize=(12, 6))
        plot_sns(y_str)
        plt.title("THIS IS TEMPORARY")
        plt.tight_layout()
        plt.savefig("plots/" + ".".join(y_str.split(r'/')) + '.pdf')
        # plt.show()

def main():
    plot_several()
    # plot_matplotlib()

if __name__ == '__main__':
    main()

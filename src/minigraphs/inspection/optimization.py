import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import numpy as np
from minigraphs.visualization import grad_plot

def plot_loss(df, targets, weights, ax=None):
        # Get axis
        if ax is None: 
            ax = plt.gca()
        
        df = df.copy()

        metrics = [f"diff_{metric}" for metric in targets.keys()]

        # Add differences
        for metric, value in targets.items():
            df[f"diff_{metric}"] = np.abs(df[f"m_{metric}"] - value)

        # Define plotting metrics
        n_iterations = df.shape[0]
        x = np.arange(n_iterations)
        cmap = plt.get_cmap('YlOrRd')
        previous_loss = 0
        for i, metric in enumerate(targets.keys()):
            column = f"diff_{metric}"
            color = 0.2 + 0.8 * i / (len(metrics) - 1)

            loss_so_far = weights[metric] * df[column]
            if i != 0:
                 loss_so_far += previous_loss

            label = "Total Loss" if i == (len(metrics)-1) else None 

            ax.fill_between(
                x,
                loss_so_far, 
                previous_loss,
                alpha=0.4,
                color=cmap(color),
                label=f"{column.split('_')[1].capitalize()}"
            )

            ax.plot(
                x,
                loss_so_far,
                linewidth=0.5,
                color=cmap(color),
                label=label
            )

            previous_loss = loss_so_far

        final_loss = df['loss'].iat[-1]
        ax.axhline(final_loss, linewidth=2.0, color='blue', linestyle=':')
        ax.text(0.50 * n_iterations, final_loss * 1.1, f'Final Loss: {final_loss:.2f}',color='blue')
        ax.set_title("Share of Total Loss Over Time")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Iteration")
        ax.legend(title="Metric")
        
def plot_losses(dfs, ax=None):
    # Get axis
    if ax is None:
        ax = plt.gca()

    # Get colormap
    cmap = plt.get_cmap('Greens')

    for i, df in enumerate(dfs):
        # Vary color map
        value = 0.20 + 0.80 * i / (len(dfs)-1)
        color = cmap(value)

        ax.plot(
            df['iteration'], 
            df['loss'],
            color=color,
            label=i
        )

    ax.legend(title="Replica")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Replica Total Loss vs. Iteration")
        
def plot_schedules(dfs, ax=None):
    if ax is None:
        ax = plt.gca()

    final_loss = [df['loss'].iat[-1] for df in dfs]
    n_dfs = len(dfs)
    max_rank = n_dfs - 1
    cmap = plt.get_cmap('cool')

    for rank, loss in enumerate(sorted(final_loss, reverse=True)):
        if rank != max_rank:
            c = 0.6 * rank / (n_dfs - 2)
            alpha = 0.5
        else: 
            c = 1.0
            alpha = 1.0

        idx = final_loss.index(loss)
        ax.plot(dfs[idx]['beta'], c=cmap(c), alpha=alpha, label=f"{loss:.2f}")

    ax.legend(title="Final Loss")
    ax.set_ylabel(r"$\beta$")
    ax.set_xlabel("Iteration")
    ax.set_title("Annealing Schedule")
        

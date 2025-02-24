import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import numpy as np
from minigraphs.visualization import grad_plot


class DisplayOptimization:
    def __init__(self, df, targets):
        # Store
        self.df = df 
        self.targets = targets

    def loss(self, ax=None):
        df = self.df.copy()

        metrics = [f"diff_{metric}" for metric in self.targets.keys()]

        # Add differences
        for metric, value in self.targets.items():
            df[f"diff_{metric}"] = np.abs(df[f"m_{metric}"] - value)

        # Define plotting metrics
        n_iterations = df.shape[0]
        x = np.arange(n_iterations)
        baseline = np.zeros(n_iterations)

        # Get axis
        if ax is None: 
            ax = plt.gca()
    
        for metric in metrics:
            data = df[metric] + baseline

            color = ax.plot(
                data,
                linewidth=0.5,
                label=metric
            )[0].get_color()

            ax.fill_between(
                x,
                data, 
                baseline,
                color=color,
                alpha=0.3
            )

            baseline = data

            ax.set_ylim([df[metrics[0]].min(), df['loss'].max()])

        ax.set_title("Total Loss Over Time")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Iteration")
        ax.legend()

        return ax

    def schedule(self, ax=None):
        if ax is None: 
            ax = plt.gca()

        ax.plot(
            self.df['beta'],
            label='schedule'
        )

        ax.set_title("Schedule")
        ax.set_ylabel("Beta")
        ax.set_xlabel("Iteration")

    def distance(self, ax=None):
        df = self.df.copy()

        # Calculate distance
        df['distance'] = 0
        for metric, value in self.targets.items():
            df['distance'] += np.square(df[f'm_{metric}'] - value)

        df['distance'] = np.sqrt(df['distance'])

        if ax is None:
            ax = plt.gca()

        ax.plot(
            df['distance']
        )

        ax.set_title("Euclidean Distance")
        ax.set_ylabel("Distance")
        ax.set_xlabel("Iteration")

    def trajectories(self, trajectory, ax=None):
        df = self.df.copy()

        cmap = plt.get_cmap('cool_r')
        norm = mcolors.Normalize(vmin=df['iteration'].min(), vmax=df['iteration'].max())
        color = lambda values: cmap(norm(values))

        # Define plane drawing function
        def draw_plane(plane):
            columns = [f"m_{name}" for name in plane]

            # Draw trajectory
            X = df[[*columns]].to_numpy()
            grad_plot(
                X[:,0],
                X[:,1],
                df['iteration'].to_numpy(),
                color,
            )

            plt.colorbar(ScalarMappable(norm=norm,cmap=cmap), ax=ax, label='Iteration')

            # Draw end points
            ax.scatter(
                x=[df.iloc[0][columns[0]]],
                y=[df.iloc[0][columns[1]]],
                color='w',
                edgecolors=['k'],
                label='Start'
            )

            ax.scatter(
                x=[df.iloc[-1][columns[0]]],
                y=[df.iloc[-1][columns[1]]],
                color='k',
                label='End',
            )

            # Draw target
            ax.scatter(
                self.targets[plane[0]],
                self.targets[plane[1]],
                color='k',
                marker='x',
                label='target'
            )

            ax.legend()
            ax.set_xlabel(plane[0])
            ax.set_ylabel(plane[1])
            ax.set_title(f"{plane[0].capitalize()}-{plane[1].capitalize()} plane")
            ax.autoscale_view()


        def draw_trajectory(trajectory):
            column = f"m_{trajectory}"

            # Draw endpoints
            style = '--'
            ax.axhline(df.iloc[0][column], linestyle=style, label='Start')
            ax.axhline(self.targets[trajectory], linestyle=style, color='r', label='Target')

            # Draw trajectory
            ax.plot(
                df['iteration'],
                df[column]
            )
            ax.set_title(f"{trajectory.capitalize()}")

            ax.legend()

        # Retrieve axis
        if ax is None:
            ax = plt.gca()

        # Draw trajectories
        if isinstance(trajectory, list) and len(trajectory) == 2:
            draw_plane(trajectory)
        else:
            draw_trajectory(trajectory)
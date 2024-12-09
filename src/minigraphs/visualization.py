import matplotlib.pyplot as plt 
import matplotlib
import matplotlib.cm as cm
import numpy as np

def plot_trajectories(df):
    fig_metrics, axes = plt.subplots(2,2,figsize=(12,6))
    fig_replicas, ax_replica = plt.subplots(figsize=(12,3))

    # Collect replicas
    replicas = df['Replica'].unique()
    cmap = cm.get_cmap('summer',len(replicas))
    cmap2 = cm.get_cmap('Set1',len(replicas))

    for replica in replicas:
        color = cmap(replica)
        idx = df['Replica'] == replica
        get_y = lambda metric : df.loc[idx,metric]
        
        # Energy plot
        y = get_y('Energy')
        x = np.arange(y.count())
        axes[0,0].plot(x,y,color=color,label=rf"$R_{replica}$")
        axes[0,0].set_xticklabels([])
        axes[0,0].set_title("Energy")
        axes[0,0].legend()
        
        # Density plot
        y = get_y('err_density')
        axes[0,1].plot(x,y,color=color)
        axes[0,1].set_xticklabels([])
        axes[0,1].set_title('Density')
        
        # Assortativity plot#
        y = get_y('err_assortativity_norm')
        axes[1,0].plot(x,y,color=color)
        axes[1,0].set_xlabel("Number of Iterations")
        axes[1,0].set_title('Assortativity (normalized)')
        
        # Clustering plot
        y = get_y('err_clustering')
        axes[1,1].plot(x,y,color=color)
        axes[1,1].set_xlabel("Number of Iterations")
        axes[1,1].set_title('Clustering')
        
        # Replica exchange
        y = get_y('Beta')
        ax_replica.plot(x,y,linewidth=10.0*(1-replica/(len(replicas)+1)),color=cmap(replica))

    maxes = df.max()
    vec = np.linspace(0,1.2,6)
    axes[0][0].set_yticks((vec * maxes['Energy']).astype(int))
    axes[0][0].set_ylabel("Absolute Percent Error")
    axes[0][0].legend(loc='upper right')
    axes[0][1].set_yticks((vec * maxes['err_density']).astype(int))
    axes[1][0].set_yticks((vec * maxes['err_assortativity_norm']).astype(int))
    axes[1][0].set_ylabel("Absolute Percent Error")
    axes[1][1].set_yticks((vec * maxes['err_clustering']).astype(int))

    ax_replica.set_ylabel(r"$\beta$")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 
import networkx as nx
from fa2 import ForceAtlas2
import sys
from scipy.sparse import load_npz



# Load the graph
adjacency_file = sys.argv[1]
drawing_file = sys.argv[2]
graph = nx.from_scipy_sparse_matrix(load_npz(adjacency_file))

# Calculate layout using force atlas
forceatlas2 = ForceAtlas2(gravity=1.0)

positions = forceatlas2.forceatlas2_networkx_layout(graph,
                                                    pos=None,
                                                    iterations=100)

# Draw Graph
fig, ax = plt.subplots(figsize=(5,5),dpi=600)
color = list(nx.clustering(graph).values())

ax.set_title(f"|V|={graph.number_of_nodes()}, den={nx.density(graph):.2e}")
ax.set_yticks([])
ax.set_xticks([])
ax.set_facecolor('black')

nx.draw_networkx_nodes(graph,positions,
                       node_size=0.1,
                       node_color=color,
                       vmin=0.0,
                       vmax=1.0,
                       alpha=0.7,
                       cmap='cool')
nx.draw_networkx_edges(graph,positions,
                       edge_color="white",
                       alpha=0.05,
                       width=0.5)

plt.savefig(drawing_file,bbox_inches='tight')
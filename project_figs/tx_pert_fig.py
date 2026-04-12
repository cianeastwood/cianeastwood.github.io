import networkx as nx
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Toggle this flag to change the bottom-right node color
is_orange = True 
# ---------------------

# 1. Define the Nodes and their specific Colors
nodes_and_colors = {
    'Teal': '#40E0D0',        
    'Purple': '#AB82FF',      
    'Pink': '#FF69B4',        
    'Light Blue': '#00BFFF', 
    'Bottom Right': 'tab:orange' if is_orange else '#000080', 
    'Magenta': '#DA70D6'     
}

# Define the edge list
edges = [
    ('Teal', 'Purple'),
    ('Teal', 'Light Blue'),
    ('Purple', 'Pink'),
    ('Purple', 'Light Blue'), 
    ('Pink', 'Light Blue'),
    ('Pink', 'Bottom Right'),
    ('Light Blue', 'Bottom Right'),
    ('Light Blue', 'Magenta')
]

# 2. Create the Graph and define Manual Positions
pos = {
    'Teal': (0.1, 0.6),        
    'Purple': (0.35, 0.9),     
    'Pink': (0.35, 0.3),       
    'Light Blue': (0.7, 0.9),  
    'Bottom Right': (0.7, 0.3),   
    'Magenta': (0.95, 0.6)     
}

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Remove the figure padding but ADD a 15% data margin to prevent node clipping
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.margins(0.025) # Changed from 0 to 0.15 to give the large nodes breathing room
ax.axis('off')
fig.patch.set_alpha(0.0) # Transparent background

# 3. Assemble and Draw the Graph Components
G = nx.Graph()
for node, color in nodes_and_colors.items():
    G.add_node(node, color=color)
G.add_edges_from(edges)

# Extract color list in graph's node order
node_colors_list = [G.nodes[node]['color'] for node in G.nodes()]

# Draw connections (Edges)
nx.draw_networkx_edges(
    G, pos,
    ax=ax,
    edge_color='#0000CD',
    width=2.5
)

# Draw distinct circular colored nodes (Nodes)
nx.draw_networkx_nodes(
    G, pos,
    ax=ax,
    node_size=3000, 
    node_color=node_colors_list,
    node_shape='o',
    edgecolors='none'
)

# Optional: Save with transparency and tight bounding box
# plt.savefig('graph.png', transparent=True, bbox_inches='tight', pad_inches=0)

plt.savefig("txpert_2.svg")
from utils import process_matchms as _myfun
import plotly.express as px
import pickle
import kmedoids
import numpy as np
from sklearn import manifold
import itertools
from plotly.colors import n_colors
import pandas as pd

# Load similarity matrices.
file = open("data/sim_matrix_ms2deepscore", 'rb')
sm = pickle.load(file) # sm = _myfun.extract_similarity_scores(sm) # <- if cosine based
file.close()
dist = 1.-sm
dist = np.round(dist, 5)

# Load Pandas Classification Table & Extract classes
file = open("data/pandas-classification-table-0012.pickle", 'rb')
class_table = pickle.load(file)
file.close()
cluster_cf = class_table["cf_class"].tolist()
cluster_cf = [str(elem).replace(" ", "_") for elem in cluster_cf] # NO WHITE SPACES ALLOWED IN CLASS STRINGS!!!

# kmedoid clustering
cluster = kmedoids.KMedoids(n_clusters=30, metric='precomputed', random_state=0, method = "fasterpam")  
cluster_km = cluster.fit_predict(dist)

# Compute T-sne embedding
model = manifold.TSNE(metric="precomputed", random_state = 123, perplexity= 15)
z = model.fit_transform(dist)
x_coords = z[:,0] * 100 # scale the coordinate space for better visualization
y_coords = z[:,1] * 100 # scale the coordinate space for better visualization
id = np.arange(0, len(x_coords), 1)
tsne_df = pd.DataFrame({"x": x_coords, "y":y_coords, "id":id, "clust": cluster})

# Create Cluster Color Dictionary
unique_clusters = np.unique(cluster)
colors = px.colors.sample_colorscale("turbo", [ n / (len(unique_clusters) - 1) for n in range(len(unique_clusters))])
colors = [ _myfun.extract_hex_from_rgb_string(elem) for elem in colors]
color_dict = {clust : colors[idx] for idx, clust in enumerate(unique_clusters)}

# Create overall figure with color_dict mapping
fig = px.scatter(tsne_df, x = "x", y = "y", color = "clust", custom_data=["id"], color_discrete_map=color_dict,
  render_mode='webgl') # <-- using efficient and highly scalable render mode; a million points easily covered.
# for more complex specifications, use go.Scattergl()
fig.update_layout(clickmode='event+select') # for lasso, box, and multi shift+click selections
# TODO: make the plot grayscale & and add color cluster on hover event. 

# Define Network Data
n_nodes = sm.shape[0]
threshold = 0.8 # TODO: implement dynamic threshold.
adj_m = _myfun.compute_adjacency_matrix(sm, threshold)
# network data packaging
all_possible_edges = list(itertools.combinations(np.arange(0, n_nodes), 2))

edges = [{'data' : {'id': str(elem[0]) + "-" + str(elem[1]), 'source': str(elem[0]),'target': str(elem[1])}} for elem in all_possible_edges if (adj_m[elem[0]][elem[1]] != 0)]
nodes = [{'data': {'id': str(elem)}, 'label': 'Node ' + str(elem),
  'position': {'x': x_coords[elem], 'y': -y_coords[elem]}, 'classes': cluster[elem]} # <-- Added -y_coord for correct orientation in line with t-sne
  for elem in np.arange(0, n_nodes)]



# , 'line-color': f"{colors[elem]}"
active_style_sheet = [{'selector' : 'edge', 'style' : {'opacity' : 0.4}}, {'selector' : 'node', 'style' : {'height' : "75%", 'width' : '75%', 'opacity' : 0.6}}]
#for elem in active_style_sheet:
#    print(elem)

# Load additional similarity matrices
file = open("data/sim_matrix_cosine_modified", 'rb')
sm2 = pickle.load(file) # sm = _myfun.extract_similarity_scores(sm) # <- if cosine based
sm2 = _myfun.extract_similarity_scores(sm2)
file.close()
file = open("data/sim_matrix_spec2vec", 'rb')
sm3 = pickle.load(file) # sm = _myfun.extract_similarity_scores(sm) # <- if cosine based
file.close()
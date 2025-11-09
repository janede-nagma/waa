import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# Load Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame
df['species'] = df.target.map(dict(enumerate(iris.target_names)))

# Create temporal dataset
np.random.seed(42)
df_temp = pd.DataFrame({
    "year": range(2015, 2025),
    "temperature": np.random.uniform(15, 30, 10)
})

# a. 1D (Linear) Data Visualization
plt.figure(figsize=(7, 4))
sns.histplot(df['sepal length (cm)'], bins=20, kde=True, color="skyblue")
plt.title("a. 1D: Distribution of Sepal Length")
plt.savefig(f"{OUTDIR}/1d_linear.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ 1D visualization saved")

# b. 2D (Planar) Data Visualization
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="sepal length (cm)", y="sepal width (cm)", hue="species")
plt.title("b. 2D: Sepal Length vs Width")
plt.savefig(f"{OUTDIR}/2d_planar.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ 2D visualization saved")

# c. 3D (Volumetric) Data Visualization
fig3d = px.scatter_3d(df, x="sepal length (cm)", y="sepal width (cm)", 
                      z="petal length (cm)", color="species",
                      title="c. 3D: Iris Features")
fig3d.write_html(f"{OUTDIR}/3d_volumetric.html")
print("✅ 3D visualization saved")

# d. Temporal Data Visualization
plt.figure(figsize=(8, 4))
sns.lineplot(data=df_temp, x="year", y="temperature", marker="o", color="orange")
plt.title("d. Temporal: Temperature Trend Over Years")
plt.savefig(f"{OUTDIR}/temporal.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Temporal visualization saved")

# e. Multidimensional Data Visualization
pair = sns.pairplot(df, hue="species", diag_kind="kde")
pair.fig.suptitle("e. Multidimensional: All Iris Features", y=1.02)
pair.savefig(f"{OUTDIR}/multidimensional.png", dpi=150)
plt.close('all')
print("✅ Multidimensional visualization saved")

# f. Tree/Hierarchical Data Visualization
X = df[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]]
linked = linkage(X, method="ward")
plt.figure(figsize=(10, 5))
dendrogram(linked, labels=df['species'].values, leaf_rotation=90, leaf_font_size=8)
plt.title("f. Tree/Hierarchical: Dendrogram of Iris Dataset")
plt.savefig(f"{OUTDIR}/hierarchical.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Hierarchical visualization saved")

# g. Network Data Visualization
G = nx.Graph()
species = df['species'].unique()
for i, sp1 in enumerate(species):
    for sp2 in species[i+1:]:
        G.add_edge(sp1, sp2, weight=np.random.randint(1, 10))

plt.figure(figsize=(6, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightgreen", 
        font_size=12, font_weight="bold", width=2)
plt.title("g. Network: Species Relationships")
plt.savefig(f"{OUTDIR}/network.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Network visualization saved")

print(f"\n🎉 All 7 visualization types saved in '{OUTDIR}/' folder")
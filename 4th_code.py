import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# Load Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame
df['species'] = df.target.map(dict(enumerate(iris.target_names)))

# 1. Histogram (1D)
plt.figure(figsize=(7, 4))
sns.histplot(df['sepal length (cm)'], bins=20, kde=True)
plt.title("1D: Sepal Length Distribution")
plt.savefig(f"{OUTDIR}/1d_histogram.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. Scatterplot (2D)
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="sepal length (cm)", y="sepal width (cm)", hue="species")
plt.title("2D: Sepal Length vs Width")
plt.savefig(f"{OUTDIR}/2d_scatter.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. Pairplot (Multidimensional)
pair = sns.pairplot(df, hue="species", diag_kind="kde")
pair.fig.suptitle("Multidimensional: Iris Features", y=1.02)
pair.savefig(f"{OUTDIR}/multidimensional_pairplot.png", dpi=150)
plt.close('all')

print(f"✅ All visualizations saved in '{OUTDIR}/'")
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors

st.set_page_config(layout="wide")

# Read the CSV file
pivot_df = pd.read_csv('03_preds_viz.csv')
pivot_df['Date'] = pd.to_datetime(pivot_df['Date'], errors='coerce')

# Define custom colormap
cmap = mcolors.LinearSegmentedColormap.from_list("n", ["red", "yellow", "green"])
boundaries = [0, 10, 25, 75, 90, 100]
colors = ["darkred", "red", "yellow", "lightgreen", "darkgreen"]
cmap = ListedColormap(colors)
norm = BoundaryNorm(boundaries, cmap.N, clip=True)

fig, ax = plt.subplots(figsize=(15, 5))  # Adjust this to fit your display

plt.xticks(fontsize=8)  # Adjust the font size
plt.yticks(fontsize=8)  # Adjust the font size

plt.title('Chance of Being Open by CORA Trail', fontsize=20, pad=20)  # Added pad=20

data = pivot_df.drop(columns=['Date'])

sns.heatmap(data, cmap=cmap, norm=norm, ax=ax, cbar_kws={'format': '%.0f%%', 'boundaries': boundaries}, annot=False)

# Get the current colorbar
cbar = ax.collections[0].colorbar

# Set the font size of the colorbar's tick labels
cbar.ax.tick_params(labelsize=8) 

for i in range(len(pivot_df['Date'])):
    for j in range(data.shape[1]):
        value = data.iloc[i, j]
        text_color = "black" if value > 20 else "white"
        plt.text(j + 0.5, i + 0.5, f'{value:.0f}%', ha='center', va='center', color=text_color, fontsize=6)  # Adjust the fontsize

plt.xticks(rotation=45)
plt.yticks(rotation=0)
ax.set_yticks(range(len(pivot_df['Date'])))
ax.set_yticklabels(pivot_df['Date'].dt.strftime('%Y-%m-%d'), fontsize=8)  # Adjust the fontsize

st.pyplot(fig, use_container_width=True)
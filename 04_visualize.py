import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors
import boto3

# Read the CSV file
pivot_df = pd.read_csv('data/03_preds_viz.csv')
del pivot_df['Milford Trails']
pivot_df['Date'] = pd.to_datetime(pivot_df['Date'], errors='coerce')

# ROUND PREDS DOWN TO THE NEXT LOWEST 5
cols = [col for col in pivot_df.columns if col != 'Date']

pivot_df[cols] = pivot_df[cols].apply(lambda x: (x // 10) * 10)

# Define custom colormap
cmap = mcolors.LinearSegmentedColormap.from_list("n", ["red", "yellow", "green"])
boundaries = [0, 10, 25, 75, 90, 100]
colors = ["darkred", "lightcoral", "lightyellow", "lightgreen", "darkgreen"]
cmap = ListedColormap(colors)
norm = BoundaryNorm(boundaries, cmap.N, clip=True)

fig, ax = plt.subplots(figsize=(15, 7))  # Adjust this to fit your display

plt.xticks(fontsize=10)  # Adjust the font size
# plt.yticks(fontsize=10)  # Adjust the font size

from datetime import datetime

today_date = datetime.today().strftime('%Y-%m-%d')
plt.title(f"Likelihood CORA Trail Open for Riding: Last Updated ({today_date})", fontsize=18, pad=20)

data = pivot_df.drop(columns=['Date'])

sns.heatmap(data, cmap=cmap, norm=norm, ax=ax, cbar_kws={'format': '%.0f%%', 'boundaries': boundaries}, annot=False)

# Get the current colorbar
cbar = ax.collections[0].colorbar

# Set the font size of the colorbar's tick labels
cbar.ax.tick_params(labelsize=8) 

for i in range(len(pivot_df['Date'])):
    for j in range(data.shape[1]):
        value = data.iloc[i, j]
        text_color = "white" if value <= 10 or value >= 90 else "black"
        plt.text(j + 0.5, i + 0.5, f'{value:.0f}%', ha='center', va='center', color=text_color, fontsize=9)  # Adjust the fontsize

plt.xticks(rotation=45)
plt.yticks(rotation=0)
ax.set_yticks(range(len(pivot_df['Date'])))
ax.set_yticklabels(pivot_df['Date'].dt.strftime('%Y-%m-%d'), fontsize=11)  # Adjust the fontsize

plt.show(block=False)


# Save the figure
filename_local = 'data/daily_trail_condition_predictions.png'
filename_s3 = 'daily_trail_condition_predictions.png'

fig.savefig(filename_local, dpi=300, bbox_inches='tight')

# Upload the image to S3
bucket_name = 'mtb-trail-condition-predictions'  # Change this to your bucket name
s3 = boto3.client('s3')

# Set the ACL to 'public-read' to allow public access
s3.upload_file(filename_local, bucket_name, filename_s3, ExtraArgs={'ACL': 'public-read'})

print("Done! Check your S3 bucket for the image.")
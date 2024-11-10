import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from flask import Flask, send_file

# Load features
features_path = './exp1/feature_conditional.json'
features = []
with open(features_path, 'r') as f:
    for line in f:
        features.append(json.loads(line.strip())[0])
features = np.array(features)
assert features.ndim == 2, "Features should be a 2D array."

# Load affinities
labels_path = './exp1/affinity.csv'
labels_df = pd.read_csv(labels_path)
affinities = labels_df['Affinity'].values
assert features.shape[0] == len(affinities), "Number of samples in features and labels do not match."

# Run t-SNE
tsne = TSNE(n_components=2, random_state=8)
features_tsne = tsne.fit_transform(features)

# Plot and save as PNG
output_path = './tsne_visualization_conditional.png'
plt.figure(figsize=(10, 7))
scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=affinities, cmap='coolwarm', s=50, alpha=0.7, edgecolors='w')
plt.title('t-SNE Visualization of High-Dimensional Features with Binding Affinity')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
cbar = plt.colorbar(scatter)
cbar.set_label('Binding Affinity')
plt.savefig(output_path)  # Save as PNG

# Set up Flask app
app = Flask(__name__)

@app.route('/')
def show_image():
    return send_file(output_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(port=8083, debug=True)

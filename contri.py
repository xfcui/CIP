import os
import numpy as np
import json
import matplotlib.pyplot as plt
from flask import Flask, send_file

file_path = './exp2/top_contri_data.json'
with open(file_path, 'r') as f:
    data = json.load(f)

output_path = os.path.join(os.getcwd(), 'graph_visualization1.png')  # Save to current working directory

def plot_graph(data, scale=10, fig_size=(10, 10), output_path=output_path):
    fig, ax = plt.subplots(figsize=fig_size)
    for edge in data['edges']:
        source = edge['source']
        target = edge['target']
        pos_source = data['positions'].get(str(source))
        pos_target = data['positions'].get(str(target))

        if pos_source and pos_target:
            edge_color = edge['color']
            if isinstance(edge_color, list):
                edge_color = tuple(edge_color)
            ax.plot(
                [pos_source[0] * scale, pos_target[0] * scale], 
                [pos_source[1] * scale, pos_target[1] * scale], 
                color=edge_color, lw=2
            )

    for node in data['nodes']:
        node_id = node['id']
        node_color = node['color']
        pos = data['positions'].get(str(node_id))

        if pos:
            ax.scatter(pos[0] * scale, pos[1] * scale, color=node_color, s=200, zorder=2)

    ax.set_aspect('equal')
    ax.axis('off')

    plt.savefig(output_path)
    plt.close(fig)
    print(f"Image saved at: {output_path}")

plot_graph(data, scale=20, fig_size=(12, 12), output_path=output_path)

app = Flask(__name__)

# Set up Flask app
app = Flask(__name__)

@app.route('/')
def show_image():
    return send_file(output_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(port=8084, debug=True)

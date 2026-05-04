import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_2d_scatter(df, x_col, y_col, label_col='label', path_col='filename', title="2D projection", variance_ratio=None):
    """
    Plots the 2D projection of the features using a pandas DataFrame.
    Maps Matplotlib collection indices to global DataFrame indices to fix pick_event bugs.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_labels = df[label_col].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    # Dictionary to map the scatter plot object (artist) to global DataFrame indices
    artist_to_indices = {}

    for label, color in zip(unique_labels, colors):
        # Create a subset for the current class
        subset = df[df[label_col] == label]
        
        sc = ax.scatter(subset[x_col], subset[y_col], 
                        c=[color], label=label, alpha=0.7, edgecolors='k', picker=5)
        
        # Store the global index of the DataFrame associated with this scatter collection
        artist_to_indices[sc] = subset.index.tolist()

    xlabel = f"Component 1 ({variance_ratio[0]*100:.2f}%)" if variance_ratio is not None else "Component 1"
    ylabel = f"Component 2 ({variance_ratio[1]*100:.2f}%)" if variance_ratio is not None else "Component 2"
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    def on_pick(event):
        sc = event.artist              # The specific scatter plot clicked
        local_ind = event.ind[0]       # Local index within the subset
        global_ind = artist_to_indices[sc][local_ind] # Mapping to global DataFrame index!
        
        # Extract correct information using the global index
        clicked_image_path = df.loc[global_ind, path_col]
        clicked_label = df.loc[global_ind, label_col]
        
        print(f"\n[{clicked_label}] -> Opening file: {clicked_image_path}")
        Image.open(clicked_image_path).show()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.tight_layout()
    plt.show()


def plot_3d_scatter(df, x_col, y_col, z_col, label_col='label', path_col='filename', title="3D projection", variance_ratio=None):
    """
    Plots the 3D projection of the features using a pandas DataFrame.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    unique_labels = df[label_col].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    artist_to_indices = {}

    for label, color in zip(unique_labels, colors):
        subset = df[df[label_col] == label]
        
        sc = ax.scatter(subset[x_col], subset[y_col], subset[z_col], 
                        c=[color], label=label, alpha=0.7, edgecolors='k', s=40, picker=5)
        
        artist_to_indices[sc] = subset.index.tolist()

    xlabel = f"Component 1 ({variance_ratio[0]*100:.2f}%)" if variance_ratio is not None else "Component 1"
    ylabel = f"Component 2 ({variance_ratio[1]*100:.2f}%)" if variance_ratio is not None else "Component 2"
    zlabel = f"Component 3 ({variance_ratio[2]*100:.2f}%)" if variance_ratio is not None else "Component 3"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.legend()

    def on_pick(event):
        sc = event.artist
        local_ind = event.ind[0] 
        global_ind = artist_to_indices[sc][local_ind]
        
        clicked_image_path = df.loc[global_ind, path_col]
        clicked_label = df.loc[global_ind, label_col]
        
        print(f"\n[{clicked_label}] -> Opening file: {clicked_image_path}")
        Image.open(clicked_image_path).show()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.tight_layout()
    plt.show()
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_2d_scatter(X_transformed, labels, image_paths, title="2D projection", variance_ratio=None):
    """
    Function that plots the 2D projection of the features.
    """
    if X_transformed.shape[1] != 2:
        raise ValueError("Data must have exactly 2 dimensions for a 2D plot.")
    
    fig = plt.figure(figsize=(10, 8))

    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        labels = np.array(labels)
        index = np.where(labels == label)
        plt.scatter(X_transformed[index, 0], X_transformed[index, 1], 
                    c=[color], label=label, alpha=0.7, edgecolors='k', picker=5)

    xlabel = f"Component 1 ({variance_ratio[0]*100:.2f}%)" if variance_ratio is not None else "Component 1"
    ylabel = f"Component 2 ({variance_ratio[1]*100:.2f}%)" if variance_ratio is not None else "Component 2"
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    def on_pick(event):
        ind = event.ind[0] 
        clicked_image_path = image_paths[ind]
        clicked_label = labels[ind]
        
        print(f"\n[{clicked_label}] -> Opening file: {clicked_image_path}")
        Image.open(clicked_image_path).show()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.tight_layout()
    plt.show()


def plot_3d_scatter(X_transformed, labels, image_paths, title="3D projection", variance_ratio=None):
    """
    Function that plots the 3D projection of the features.
    """
    if X_transformed.shape[1] != 3:
        raise ValueError("Data must have exactly 3 dimensions for a 3D plot.")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        labels = np.array(labels)
        index = np.where(labels == label)
        ax.scatter(X_transformed[index, 0], X_transformed[index, 1], X_transformed[index, 2], 
                   c=[color], label=label, alpha=0.7, edgecolors='k', s=40, picker=5)

    xlabel = f"Component 1 ({variance_ratio[0]*100:.2f}%)" if variance_ratio is not None else "Component 1"
    ylabel = f"Component 2 ({variance_ratio[1]*100:.2f}%)" if variance_ratio is not None else "Component 2"
    zlabel = f"Component 3 ({variance_ratio[2]*100:.2f}%)" if variance_ratio is not None else "Component 3"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.legend()

    def on_pick(event):
        ind = event.ind[0] 
        clicked_image_path = image_paths[ind]
        clicked_label = labels[ind]
        
        print(f"\n[{clicked_label}] -> Opening file: {clicked_image_path}")
        Image.open(clicked_image_path).show()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.tight_layout()
    plt.show()
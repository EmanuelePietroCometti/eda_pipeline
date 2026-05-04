from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
import umap

def build_pca_pipeline(n_components=2, random_state=42):
    """
    Function that build PCA pipeline with the scikit-learn library
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=random_state))
    ])

def build_tsne_pipeline(n_components=2, pre_pca_dims=50, random_state=42):
    """
    Function that build a tsne pipeline with the scikit-learn library 
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pre_pca", PCA(n_components=pre_pca_dims, random_state=random_state)),
        ("tsne", TSNE(n_components=n_components, random_state=random_state, init="pca", learning_rate="auto"))
    ])

def build_umap_pipeline(n_components=2, random_state=42):
    """
    Function that build a UMAP pipeline with the scikit-learn library 
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("umap", umap.UMAP(n_components=n_components, random_state=random_state))
    ])



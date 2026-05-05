from src.feature_extraction import extract_features
from src.eda import build_pca_pipeline, build_tsne_pipeline, build_umap_pipeline
from src.visualizzation import plot_2d_scatter, plot_3d_scatter
import numpy as np

def main():
    # Extract features, labels, and file paths
    data = extract_features()
    X_features = np.stack(data['features'].values)


    # PCA Dimensionality Reduction
    pca_pipeline = build_pca_pipeline(n_components=0.95)
    X_pca_95 = pca_pipeline.fit_transform(X_features)
    pca_model = pca_pipeline.named_steps["pca"]
    n_95 = pca_model.n_components_
    variance_ratio = pca_model.explained_variance_ratio_

    data['pca_x'] = X_pca_95[:, 0]
    data['pca_y'] = X_pca_95[:, 1]
    if X_pca_95.shape[1] >= 3:
        data['pca_z'] = X_pca_95[:, 2]

    plot_2d_scatter(
        df=data,
        x_col='pca_x',
        y_col='pca_y',
        title="PCA 2D",
        variance_ratio=variance_ratio[:2]
    )

    if 'pca_z' in data.columns:
        plot_3d_scatter(
            df=data,
            x_col='pca_x',
            y_col='pca_y',
            z_col='pca_z',
            title="PCA 3D",
            variance_ratio=variance_ratio[:3] 
        )

    # t-SNE Dimensionality Reduction
    tsne_pipeline = build_tsne_pipeline(n_components=2, pre_pca_dims=n_95)
    X_tsne_2d = tsne_pipeline.fit_transform(X_features)

    data['tsne_x'] = X_tsne_2d[:, 0]
    data['tsne_y'] = X_tsne_2d[:, 1]

    plot_2d_scatter(
        df=data,
        x_col='tsne_x',
        y_col='tsne_y',
        title="TSNE 2D"
    )

    n_samples = X_features.shape[0]
    perplexity = min(50.0, max(5.0, n_samples / 3.0))
    tsne_pipeline_3d = build_tsne_pipeline(n_components=3, pre_pca_dims=n_95, perplexity=perplexity, early_exaggeration=15.0, n_iter=1500)
    X_tsne_3d = tsne_pipeline_3d.fit_transform(X_features)

    data['tsne_x_3d'] = X_tsne_3d[:, 0]
    data['tsne_y_3d'] = X_tsne_3d[:, 1]
    data['tsne_z_3d'] = X_tsne_3d[:, 2]

    plot_3d_scatter(
        df=data,
        x_col='tsne_x_3d',
        y_col='tsne_y_3d',
        z_col='tsne_z_3d',
        title="TSNE 3D"
    )

    # UMAP Dimensionality Reduction
    umap_pipeline = build_umap_pipeline(n_components=2)
    X_umap_2d = umap_pipeline.fit_transform(X_features)

    data['umap_x'] = X_umap_2d[:, 0]
    data['umap_y'] = X_umap_2d[:, 1]

    plot_2d_scatter(
        df=data,
        x_col='umap_x',
        y_col='umap_y',
        title="UMAP 2D"
    )

    umap_pipeline_3d = build_umap_pipeline(n_components=3) 
    X_umap_3d = umap_pipeline_3d.fit_transform(X_features)

    data['umap_x_3d'] = X_umap_3d[:, 0]
    data['umap_y_3d'] = X_umap_3d[:, 1]
    data['umap_z_3d'] = X_umap_3d[:, 2]

    plot_3d_scatter(
        df=data,
        x_col='umap_x_3d',
        y_col='umap_y_3d',
        z_col='umap_z_3d',
        title="UMAP 3D"
    )

if __name__ == "__main__":
    main()
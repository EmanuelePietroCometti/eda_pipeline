from src.feature_extraction import extract_features
from src.eda import build_pca_pipeline, build_tsne_pipeline, build_umap_pipeline
from src.visualizzation import plot_2d_scatter, plot_3d_scatter

def main():
    # Extract features, labels, and file paths
    features, labels, paths = extract_features()

    # PCA Dimensionality Reduction
    pca_pipeline = build_pca_pipeline(n_components=0.95)
    X_pca_95 = pca_pipeline.fit_transform(features)
    pca_model = pca_pipeline.named_steps["pca"]
    n_95 = pca_model.n_components_

    plot_2d_scatter(
        X_transformed=X_pca_95[:, :2],
        labels=labels,
        image_paths=paths,
        title="PCA 2D",
        variance_ratio=pca_model.explained_variance_ratio_[:2]
    )

    plot_3d_scatter(
        X_transformed=X_pca_95[:, :3],
        labels=labels,
        image_paths=paths,
        title="PCA 3D",
        variance_ratio=pca_model.explained_variance_ratio_[:3] 
    )

    # t-SNE Dimensionality Reduction
    tsne_pipeline = build_tsne_pipeline(n_components=2, pre_pca_dims=n_95)
    X_tsne_2d = tsne_pipeline.fit_transform(features)

    plot_2d_scatter(
        X_transformed=X_tsne_2d,
        labels=labels,
        image_paths=paths,
        title="TSNE 2D"
    )

    tsne_pipeline_3d = build_tsne_pipeline(n_components=3, pre_pca_dims=n_95)
    X_tsne_3d = tsne_pipeline_3d.fit_transform(features)

    plot_3d_scatter(
        X_transformed=X_tsne_3d,
        labels=labels,
        image_paths=paths,
        title="TSNE 3D"
    )

    # UMAP Dimensionality Reduction
    umap_pipeline = build_umap_pipeline(n_components=2)
    X_umap_2d = umap_pipeline.fit_transform(features)

    plot_2d_scatter(
        X_transformed=X_umap_2d,
        labels=labels,
        image_paths=paths,
        title="UMAP 2D"
    )

    umap_pipeline_3d = build_umap_pipeline(n_components=3) 
    X_umap_3d = umap_pipeline_3d.fit_transform(features)

    plot_3d_scatter(
        X_transformed=X_umap_3d,
        labels=labels,
        image_paths=paths,
        title="UMAP 3D"
    )

if __name__ == "__main__":
    main()
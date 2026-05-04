# Technical documentation: Exploratory Data Analysis (EDA) pipeline for images

This document provides an in-depth guide to the functioning of the exploratory data analysis (EDA) pipeline implemented in the repository. The pipeline integrates Deep Learning techniques for feature extraction and advanced dimensionality reduction algorithms for data visualization and interpretation.

## 1. Pipeline architecture
The system is designed to transform raw images into meaningful and visualizable vector representations through a two-stage process within the repository:

### 1.1 Feature extraction
Extraction occurs in the `src/feature_extraction.py` module.
* **Backbone Model**: A pre-trained model (default `wide_resnet50_2`) loaded via the `torchvision.models` library is used.
* **Head Removal**: The final classification layer is removed to obtain the feature vector (embedding) before the actual classification takes place.
* **Preprocessing**: Images are resized (default 256x256), normalized according to ImageNet standards, and loaded via `DataLoader` with GPU support to maximize processing speed.

### 1.2 Dimensionality reduction
In this stage, high-dimensional vectors are projected into a 2D or 3D space. The reduction pipelines are defined within the `src/eda.py` module and use the `scikit-learn` and `umap-learn` libraries.

---

## 2. Analysis of the algorithms used
The three implemented algorithms perform distinct mathematical transformations to reduce data dimensionality. 

### 2.1 PCA (Principal Component Analysis)
PCA is a linear technique that identifies the directions that maximize data variance.
* **Functioning**: It calculates the covariance matrix of the centered data and extracts its eigenvectors.
* **Configuration**: The implemented pipeline uses `n_components=0.95`, automatically calculating the exact number of components needed to explain at least 95% of the total variance of the original data.
* **Insights**: PCA allows us to evaluate the global shape and initial linear separability of the dataset. If the classes (e.g., "Good" vs "Reject") are cleanly separated in the PCA plot, we can deduce that the features are highly discriminative and a simple, fast linear classifier will suffice in production. It is also excellent for spotting macroscopic anomalies.

### 2.2 t-SNE (t-Distributed Stochastic Neighbor Embedding)
t-SNE is a non-linear probabilistic algorithm focused on preserving local structures.
* **Functioning**: It converts Euclidean distances into conditional probabilities (using a Gaussian distribution in high dimensionality and a Student's t-distribution in low dimensionality) and minimizes the Kullback-Leibler divergence.
* **Optimization**: To handle the extremely high initial dimensionality and avoid performance bottlenecks, the data is first compressed by a preliminary PCA phase before passing to the `TSNE` algorithm.
* **Insights**: t-SNE is ideal for microscopic exploration and **revealing non-linear separability**. If classes appear as a mixed blob in PCA but form perfectly separated clusters in t-SNE, we can deduce that the features are effective, but the decision boundary is highly complex. This dictates that a non-linear classifier (like Random Forest, SVM with RBF, or a Neural Network) will be required. Furthermore, it allows us to deduce the presence of hidden sub-clusters (e.g., if a single "Reject" class splits into distinct islands, the algorithm has found different typologies of defects based on visual similarity).

### 2.3 UMAP (Uniform Manifold Approximation and Projection)
UMAP is based on Riemannian geometry and algebraic topology.
* **Functioning**: It optimizes a Cross-Entropy metric to compare topological distributions.
* **Advantages**: It manages to better preserve not only the local structure (clusters) but also the global structure (relative distances between different clusters), making it computationally faster and more scalable.
* **Insights**: UMAP provides a balanced view by **handling non-linear separability** while also preserving global topology. Like t-SNE, it reveals complex non-linear boundaries where linear methods fail, but by preserving global distances better, it allows us to deduce hierarchical relationships and defect severity. If a "Minor Defect" cluster is positioned between the "Good" cluster and the "Severe Defect" cluster, we can infer a continuous, non-linear progression of degradation.

---

## 3. Configuration and usage guide
The behavior of the pipeline is entirely controlled by the `config.yaml` file.

### 3.1 Dependencies installation and commands to run the pipeline
First of all, you need the **Poetry** dependency manager to run this project. You can download the dependency manager following the official Poetry installation guide. 
After installing Poetry, you can run the following command to install the project's dependencies:
```bash
poetry install
```
Hardware Note: It is highly recommended to use a machine equipped with a CUDA-compatible NVIDIA GPU to drastically accelerate the feature extraction phase via PyTorch.

At this point, after the project configuration (see section 3.2), you can run the program with the following command:
```bash
python main.py
```

### 3.2 Structure of the `config.yaml` file
The main configuration file allows you to set fundamental parameters without having to modify the source code:
* **`general_configuration`**: Specifies the target dimensions to scale the images (`img_size`) and which formats to read via the `valid_extensions` list.
* **`features_extractor`**: Defines the model to use (`backbone`) and the input folders via the `folder_map` mapping.

### 3.3 Folder organization
The code is highly flexible and does not require the data to be located inside the project folder. It is recommended to structure your images in directories divided by category within your file system and insert their absolute paths into the `folder_map`:
```yaml
# config.yaml configuration example
features_extractor:
  folder_map:
    'C:/percorso/dati/good': "Good"
    'C:/percorso/dati/reject': "Reject"
```
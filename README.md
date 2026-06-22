# ADLM

**Segmentation and Classification Tasks on Gaussian Representations of Images**

This is the main repository of our team for the Applied Deep Learning in Medicine (ADLM) course, Summer Semester 2026.

**Team Members:** Maximilian Böhmichen, Dominik Hack, Yiyang Qian  
**Supervisor:** Nil Stolt-Ansó  
**Institution:** Technische Universität München (TUM), TUM AIM Lab  
**Date:** June 24th, 2026

---

## Project Roadmap

Here is the current timeline and progress of our semester project:

<img width="955" height="318" alt="image" src="https://github.com/user-attachments/assets/fd1e395a-525f-45ca-8d03-8e776f4cd0f0" />


---

## Motivation & Problem Definition

* **The Voxel Explosion:** Moving to high-resolution images or 3D/4D grid data causes the number of voxels to grow exponentially.
* **Hardware Limitations:** Standard grid-based architectures like CNNs and Transformers operate directly on pixels. When applied to massive 3D structures, their memory requirements explode.
* **The Compression Dilemma:** Naive downsampling leads to severe information loss. Therefore, highly efficient, non-destructive compression methods are critically needed.

---

## Proposed Methodology

### 1. Why Gaussians?
To solve the compression bottleneck, we utilize **Gaussian representations**. Transitioning an image into a point-cloud of Gaussians is highly memory-efficient because the number of Gaussians required is significantly smaller than the total number of pixels. 

<img width="953" height="175" alt="image" src="https://github.com/user-attachments/assets/e19fa4ba-473a-41b8-8487-b19e037b8d06" />


### 2. Graph Neural Networks (GNNs)
* **Off-Grid Architecture:** Because Gaussian point clouds are unstructured and no longer fit into a regular grid, traditional CNNs cannot be used. We propose using **Graph Neural Networks (GNNs)**, which are the standard standard for processing off-grid point clouds.
* **Graph Construction:** We treat individual Gaussian parameters as input node features and feed them directly into the GNN. The pipeline learns structural topologies using Vertex (node) embeddings, Edge (link) attributes/embeddings, and Global (master node) embeddings.

<img width="1188" height="363" alt="image" src="https://github.com/user-attachments/assets/b75c00ea-4c56-4cbb-906d-4aba91e2a951" />


### 3. GNN vs. ResNet-8 Architecture Design
To ensure a rigorous and fair evaluation, our **2DGS + GNN** model is explicitly architected to maintain a comparable parameter count (~78k) to our baseline ResNet-8 model.

* **ResNet-8 Baseline:** Implements a standard Conv Stem ($3\times3$ Conv + BN + ReLU) followed by multiple convolutional blocks, Global Average Pooling, and a Fully Connected Classifier.
* **Our 2DGS + GNN:** Converts data into a 2D Gaussian Graph, projects features ($Linear + LN + ReLU$), passes them through successive Graph Convolution blocks with Residual connections, and applies Global Mean Pooling before the final Classifier. Inside each Graph Conv, we employ Linear layers, ReLU, LayerNorm, and Mean aggregation.
* <img width="882" height="491" alt="image" src="https://github.com/user-attachments/assets/a24c9403-8bf5-4d14-8a11-a8da5803216f" />


  

---

## Preliminary Performance Results (AUC)

We evaluated classification performance across multiple medical datasets using the Area Under the Curve (AUC) metric.

<img width="844" height="442" alt="image" src="https://github.com/user-attachments/assets/4531c3fc-c44c-4147-ad6c-41db37614d5d" />


* **OrganMNIST3D (3D):** Our GNN achieved a striking **AUC of 0.96**, performing neck-and-neck with the computationally heavy 3D ResNet8 baseline (0.99) while operating on a drastically reduced memory footprint.
* **ChestMNIST (2D):** Our GNN reached an AUC of 0.72. While slightly behind the grid-based ResNet8 (0.78), it significantly outclassed alternative off-grid representation methods like `inr2vec` (0.62).

---

## 🧠 Discussion: Why Not Implicit Neural Representations (INRs)?

Implicit Neural Representations (INRs) represent another prominent off-grid paradigm where images are parameterized as continuous functions mapped inside an MLP (e.g., using SIREN or ReLU with Positional Encodings). 

However, INRs present severe foundational traps for downstream classification tasks:
1. **No Spatial Locality:** An INR coordinates are melted directly into a chaotic soup of raw network weights. 
2. **Geometric DL Failure:** Without physical coordinate lists or explicit geometric structuring, standard geometric deep learning models completely fail.
3. **Dimensionality Explosion:** Classifying an INR requires naively flattening all network parameters into an immense, unorganized input vector.

While approaches like **`inr2vec`** attempt to bridge this by utilizing a Weight-Space Encoder and Max Pooling to compress chaotic weights into a fixed-size latent embedding, our empirical benchmarks prove that **directly mapping Gaussian parameters into structural Graph Neural Networks yields vastly superior representation capability** (scoring 0.72 AUC vs. `inr2vec`'s 0.62 AUC on ChestMNIST).

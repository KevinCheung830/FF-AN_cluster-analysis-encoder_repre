# S6207 Assignment 4: Autoencoder with Clustering Analysis

This project implements a feedforward autoencoder on the Titanic dataset, compares different activation functions, and performs clustering analysis on the encoded representations.

## 📋 Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Basic Tasks](#basic-tasks)
- [Advanced Tasks](#advanced-tasks)
- [Key Functions & Classes](#key-functions--classes)
- [Results Summary](#results-summary)

## 🎯 Overview

The project consists of two main parts:
1. **Basic Tasks**: Train a feedforward autoencoder on the Titanic dataset
2. **Advanced Tasks**: Compare activation functions and perform clustering analysis

## 🚀 Setup

### Prerequisites

- Python 3.12+
- UV package manager

### Installation

```bash
# Install dependencies using UV
uv sync
```

### Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `torch` - PyTorch for neural networks
- `scikit-learn` - Preprocessing and clustering
- `matplotlib` - Visualization
- `scipy` - Statistical analysis
- `seaborn` - Enhanced visualizations

## 📚 Basic Tasks

### Step 1: Data Loading and Preprocessing

**Key Operations:**
- Load Titanic dataset from `data.txt`
- **Handle missing values**:
  - `Age`: Fill with median
  - `Fare`: Fill with median
  - `Embarked`: Fill with mode
  - `Cabin`: Fill with 'Unknown'
- **Feature selection**: Use `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Cabin`, `Embarked`
- **One-hot encoding**: Convert categorical variables (`Pclass`, `Sex`, `Cabin`, `Embarked`)
- **Feature scaling**: Apply `StandardScaler` to normalize features

**Key Function:**
```python
pd.get_dummies(df_features, columns=categorical_cols, drop_first=False)
StandardScaler().fit_transform(df_encoded)
```

### Step 2: PyTorch DataLoader Setup

**Key Operations:**
- Create custom `TitanicDataset` class extending `torch.utils.data.Dataset`
- **80/20 train/test split** using `random_split`
- Create `DataLoader` instances with batch size 32

**Key Class:**
```python
class TitanicDataset(Dataset):
    def __init__(self, data):
        self.data = torch.FloatTensor(data.values)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
```

**Key Function:**
```python
random_split(dataset, [train_size, test_size])
DataLoader(dataset, batch_size=32, shuffle=True/False)
```

### Step 3: Autoencoder Architecture

**Key Architecture:**
- **Encoder**: Input → 64 → 32 → 8 (latent dimension)
- **Decoder**: 8 (latent) → 32 → 64 → Input
- **Activation**: ReLU
- **Latent dimension**: 8

**Key Class:**
```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        # Encoder: input_dim → 64 → 32 → latent_dim
        # Decoder: latent_dim → 32 → 64 → input_dim
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

### Step 4: Training

**Key Operations:**
- **Loss function**: MSE Loss (`nn.MSELoss()`)
- **Optimizer**: Adam with learning rate 0.001
- **Training epochs**: 60 (>50 as required)
- Track train and test losses

**Key Functions:**
```python
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss.backward()
optimizer.step()
```

### Step 5: Evaluation

**Key Metrics:**
- Final Test MSE Loss
- Final Test RMSE
- Training history visualization

## 🔬 Advanced Tasks

### Task 1: Activation Function Comparison

**Key Operations:**
- Train autoencoders with **5 different activation functions**:
  - `ReLU`
  - `Tanh`
  - `Sigmoid`
  - `LeakyReLU` (alpha=0.2)
  - `ELU`
- Compare training/test losses across all activations
- Identify best activation function

**Key Class:**
```python
class FlexibleAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8, activation='relu'):
        # Selects activation function dynamically
        # Supports: 'relu', 'tanh', 'sigmoid', 'leakyrelu', 'elu'
    
    def encode(self, x):
        """Extract encoded representation"""
        return self.encoder(x)
```

**Key Function:**
```python
best_activation = results_df.loc[results_df['Test Loss'].idxmin(), 'Activation']
```

### Task 2: Clustering Analysis

**Key Operations:**
- Extract encoded representations from trained models
- Use encoded data from **best activation function**
- Determine optimal number of clusters using:
  - **Elbow method** (inertia)
  - **Silhouette score** (higher is better)
  - **Davies-Bouldin score** (lower is better)

**Key Functions:**
```python
model.encode(batch)  # Extract latent representations
silhouette_score(data, labels)
davies_bouldin_score(data, labels)
```

### Task 3: Apply Multiple Clustering Algorithms

**Key Algorithms:**
1. **K-Means Clustering**
   - Partition-based clustering
   - Requires pre-specified number of clusters
   - Fast and efficient

2. **DBSCAN Clustering**
   - Density-based clustering
   - Handles noise points (outliers)
   - Discovers clusters of arbitrary shape

3. **Hierarchical/Agglomerative Clustering**
   - Builds cluster hierarchy
   - Uses linkage criteria
   - Produces dendrogram structure

**Key Functions:**
```python
KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
DBSCAN(eps=0.5, min_samples=5)
AgglomerativeClustering(n_clusters=optimal_k)
```

### Task 4: Cluster Analysis and Comparison

**Key Operations:**
- **Analyze cluster characteristics**:
  - Cluster sizes and distributions
  - Survival rates by cluster
  - Pclass distributions
  - Age statistics
- **Visualize clusters** using PCA (2D projection)
- **Compare algorithms** using:
  - Silhouette Score
  - Davies-Bouldin Score
  - Adjusted Rand Index (for algorithm agreement)

**Key Functions:**
```python
PCA(n_components=2).fit_transform(encoded_data)
adjusted_rand_score(labels1, labels2)
pd.crosstab(labels1, labels2)  # Contingency table
```

### Task 5: Comprehensive Report Generation

**Key Outputs:**
- Summary of basic task results
- Activation function comparison table
- Clustering algorithm performance metrics
- Key findings and interpretations
- Visual summary dashboard

**Key Visualizations:**
- Training curves for all activation functions
- Cluster visualizations (PCA 2D)
- Survival rate analysis by cluster
- Age distribution by cluster

## 🔑 Key Functions & Classes

### Data Processing

| Function/Class | Purpose |
|---------------|---------|
| `TitanicDataset` | Custom PyTorch Dataset for Titanic data |
| `StandardScaler` | Normalize features to zero mean and unit variance |
| `pd.get_dummies()` | One-hot encode categorical variables |

### Model Architecture

| Class | Purpose |
|-------|---------|
| `Autoencoder` | Basic feedforward autoencoder with ReLU |
| `FlexibleAutoencoder` | Autoencoder with configurable activation function |
| `nn.Sequential()` | Stack layers in encoder/decoder |
| `nn.Linear()` | Fully connected layer |
| `nn.ReLU()`, `nn.Tanh()`, etc. | Activation functions |

### Training

| Function | Purpose |
|---------|---------|
| `nn.MSELoss()` | Mean Squared Error loss function |
| `optim.Adam()` | Adam optimizer |
| `model.train()` | Set model to training mode |
| `model.eval()` | Set model to evaluation mode |
| `loss.backward()` | Compute gradients |
| `optimizer.step()` | Update model parameters |

### Clustering

| Function | Purpose |
|---------|---------|
| `KMeans()` | K-means clustering algorithm |
| `DBSCAN()` | Density-based clustering |
| `AgglomerativeClustering()` | Hierarchical clustering |
| `silhouette_score()` | Measure clustering quality |
| `davies_bouldin_score()` | Measure cluster separation |
| `adjusted_rand_score()` | Compare cluster assignments |

### Visualization

| Function | Purpose |
|---------|---------|
| `PCA()` | Dimensionality reduction for visualization |
| `plt.plot()` | Plot training curves |
| `plt.scatter()` | Visualize clusters |
| `plt.bar()` | Compare metrics |

## 📊 Results Summary

### Basic Task Results
- **Dataset**: Titanic (891 samples)
- **Input dimension**: Varies based on one-hot encoding (~100+ features)
- **Latent dimension**: 8
- **Training epochs**: 60
- **Final Test MSE Loss**: ~0.845
- **Final Test RMSE**: ~0.920

### Activation Function Comparison
- **Best activation**: Varies (typically ReLU or ELU)
- **Test loss range**: Typically 0.75 - 0.90
- All activations show similar performance with slight variations

### Clustering Results
- **Optimal clusters**: Typically 3-5 (determined by silhouette score)
- **Best algorithm**: Varies (often K-Means or Hierarchical)
- **Silhouette scores**: Typically 0.2 - 0.4
- Clusters show meaningful patterns in survival rates and demographics

## 📝 Usage

1. **Run the notebook**:


2. **Execute cells sequentially**:
   - Cells 1-9: Basic tasks
   - Cells 10-23: Advanced tasks

3. **View results**:
   - Training curves and metrics are printed
   - Visualizations are displayed inline
   - Final report is generated automatically

## 📌 Notes

- **Reproducibility**: Random seeds are set (42) for consistent results
- **Data split**: 80% training, 20% testing
- **Batch size**: 32
- **Learning rate**: 0.001
- **Missing values**: Handled with median/mode imputation
- **Feature scaling**: Critical for autoencoder performance

## 🔍 Key Insights

1. **Overfitting**: Train loss typically lower than test loss, indicating some overfitting
2. **Activation functions**: ReLU and ELU generally perform best for this task
3. **Clustering**: Encoded representations reveal meaningful passenger groups
4. **Survival patterns**: Clusters show distinct survival rate variations
5. **Algorithm comparison**: Different clustering algorithms produce similar but not identical results

---

**Author**: S6207 Assignment 4  
**Date**: 2025  
**Dataset**: Titanic Passenger Survival


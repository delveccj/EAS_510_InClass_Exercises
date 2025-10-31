#!/usr/bin/env python3
"""
MNIST PCA Comparison Demo
Compares Standard PCA, Randomized PCA, and Incremental PCA on MNIST digits
Shows 2D projections, reconstruction quality, and performance metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
import time

# Load MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.to_numpy() / 255.0  # Normalize to [0, 1]
y = mnist.target.to_numpy()

# Use subset for faster computation (optional: remove [:10000] for full dataset)
X_sample = X[:10000]
y_sample = y[:10000]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

# Define dimensions to test
n_components_list = [2, 10, 50, 100, 200]

# Dictionary to store results
results = {}

# Standard PCA
print("\nRunning Standard PCA...")
for n_comp in n_components_list:
    start_time = time.time()
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_pca)
    elapsed = time.time() - start_time
    
    results[f'standard_{n_comp}'] = {
        'transformed': X_pca,
        'reconstructed': X_reconstructed,
        'explained_variance': pca.explained_variance_ratio_.sum(),
        'time': elapsed,
        'pca_obj': pca
    }
    print(f"  {n_comp} components: {elapsed:.3f}s, variance explained: {pca.explained_variance_ratio_.sum():.3f}")

# Randomized PCA
print("\nRunning Randomized PCA...")
for n_comp in n_components_list:
    start_time = time.time()
    pca = PCA(n_components=n_comp, svd_solver='randomized')
    X_pca = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_pca)
    elapsed = time.time() - start_time
    
    results[f'randomized_{n_comp}'] = {
        'transformed': X_pca,
        'reconstructed': X_reconstructed,
        'explained_variance': pca.explained_variance_ratio_.sum(),
        'time': elapsed,
        'pca_obj': pca
    }
    print(f"  {n_comp} components: {elapsed:.3f}s, variance explained: {pca.explained_variance_ratio_.sum():.3f}")

# Incremental PCA (Gaussian-like, works well for large datasets)
print("\nRunning Incremental PCA...")
for n_comp in n_components_list:
    start_time = time.time()
    ipca = IncrementalPCA(n_components=n_comp, batch_size=200)
    X_pca = ipca.fit_transform(X_scaled)
    X_reconstructed = ipca.inverse_transform(X_pca)
    elapsed = time.time() - start_time
    
    results[f'incremental_{n_comp}'] = {
        'transformed': X_pca,
        'reconstructed': X_reconstructed,
        'explained_variance': ipca.explained_variance_ratio_.sum(),
        'time': elapsed,
        'pca_obj': ipca
    }
    print(f"  {n_comp} components: {elapsed:.3f}s, variance explained: {ipca.explained_variance_ratio_.sum():.3f}")

# Visualization 1: 2D scatter plot comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
methods = ['standard', 'randomized', 'incremental']
titles = ['Standard PCA', 'Randomized PCA', 'Incremental PCA']

for idx, (method, title) in enumerate(zip(methods, titles)):
    X_2d = results[f'{method}_2']['transformed']
    scatter = axes[idx].scatter(X_2d[:, 0], X_2d[:, 1], 
                                c=y_sample.astype(int), 
                                cmap='tab10', 
                                alpha=0.6, 
                                s=10)
    axes[idx].set_title(f'{title}\n2 Components')
    axes[idx].set_xlabel('First Component')
    axes[idx].set_ylabel('Second Component')
    
plt.colorbar(scatter, ax=axes, label='Digit Class')
plt.tight_layout()
plt.savefig('images/pca_2d_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualization 2: Reconstruction quality
fig, axes = plt.subplots(len(n_components_list), 4, figsize=(16, 12))
sample_idx = 0  # First digit in dataset

for row, n_comp in enumerate(n_components_list):
    # Original
    axes[row, 0].imshow(X_sample[sample_idx].reshape(28, 28), cmap='gray')
    axes[row, 0].set_title('Original' if row == 0 else '')
    axes[row, 0].axis('off')
    
    # Reconstructions
    for col, method in enumerate(['standard', 'randomized', 'incremental'], 1):
        reconstructed = scaler.inverse_transform(results[f'{method}_{n_comp}']['reconstructed'])
        axes[row, col].imshow(reconstructed[sample_idx].reshape(28, 28), cmap='gray')
        if row == 0:
            axes[row, col].set_title(titles[col-1])
        axes[row, col].axis('off')
    
    axes[row, 0].set_ylabel(f'{n_comp} components', rotation=0, labelpad=50, va='center')

plt.tight_layout()
plt.savefig('images/pca_reconstruction_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualization 3: Explained variance comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart of explained variance
n_comps = [str(n) for n in n_components_list]
width = 0.25
x = np.arange(len(n_comps))

for idx, method in enumerate(['standard', 'randomized', 'incremental']):
    variances = [results[f'{method}_{n}']['explained_variance'] for n in n_components_list]
    axes[0].bar(x + idx*width, variances, width, label=titles[idx])

axes[0].set_xlabel('Number of Components')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Explained Variance by Method')
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(n_comps)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Bar chart of computation time
for idx, method in enumerate(['standard', 'randomized', 'incremental']):
    times = [results[f'{method}_{n}']['time'] for n in n_components_list]
    axes[1].bar(x + idx*width, times, width, label=titles[idx])

axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Time (seconds)')
axes[1].set_title('Computation Time by Method')
axes[1].set_xticks(x + width)
axes[1].set_xticklabels(n_comps)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('images/pca_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
for n_comp in n_components_list:
    print(f"\n{n_comp} Components:")
    for method, title in zip(['standard', 'randomized', 'incremental'], titles):
        key = f'{method}_{n_comp}'
        print(f"  {title:20s}: {results[key]['time']:.3f}s, "
              f"Variance: {results[key]['explained_variance']:.3f}")

print("\nVisualizations saved:")
print("  - pca_2d_comparison.png")
print("  - pca_reconstruction_comparison.png")
print("  - pca_metrics_comparison.png")
#!/usr/bin/env python3
"""
PCA Image Encoder
Uses randomized PCA to encode/reconstruct 10 MNIST images and saves them as PNG files
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Create output directory
output_dir = 'images/pca_reconstructed_images'
os.makedirs(output_dir, exist_ok=True)

print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.to_numpy() / 255.0  # Normalize to [0, 1]
y = mnist.target.to_numpy()

# Select 10 different digits (one of each class 0-9)
selected_images = []
selected_labels = []
used_digits = set()

for i in range(len(X)):
    digit = int(y[i])
    if digit not in used_digits:
        selected_images.append(X[i])
        selected_labels.append(digit)
        used_digits.add(digit)
        if len(selected_images) == 10:
            break

selected_images = np.array(selected_images)
print(f"Selected digits: {selected_labels}")

# Standardize the full dataset for PCA training
print("Training PCA on dataset...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[:10000])  # Use subset for training

# Test different PCA component counts
component_counts = [10, 50, 100, 200]

for n_components in component_counts:
    print(f"\nProcessing with {n_components} components...")
    
    # Create randomized PCA
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    pca.fit(X_scaled)
    
    # Transform and reconstruct our selected images
    selected_scaled = scaler.transform(selected_images)
    encoded = pca.transform(selected_scaled)
    reconstructed_scaled = pca.inverse_transform(encoded)
    reconstructed = scaler.inverse_transform(reconstructed_scaled)
    
    # Clip to [0, 1] range
    reconstructed = np.clip(reconstructed, 0, 1)
    
    # Save each reconstructed image
    component_dir = os.path.join(output_dir, f'{n_components}_components')
    os.makedirs(component_dir, exist_ok=True)
    
    # Save the PCA "key" (model + encoded vectors)
    pca_data = {
        'pca_model': pca,
        'scaler': scaler,
        'encoded_vectors': encoded,
        'original_images': selected_images,
        'labels': selected_labels,
        'n_components': n_components,
        'explained_variance': pca.explained_variance_ratio_.sum(),
        'compression_ratio': 784 / n_components  # 28x28=784 pixels -> n_components
    }
    
    # Save as pickle file
    key_file = os.path.join(component_dir, 'pca_key.pkl')
    with open(key_file, 'wb') as f:
        pickle.dump(pca_data, f)
    
    print(f"  PCA key saved: {key_file}")
    print(f"  Compression: 784 → {n_components} ({pca_data['compression_ratio']:.1f}x smaller)")
    
    for i, (img, label) in enumerate(zip(reconstructed, selected_labels)):
        # Save as PNG
        plt.figure(figsize=(2, 2))
        plt.imshow(img.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        
        filename = f'digit_{label}_idx_{i}.png'
        filepath = os.path.join(component_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
    
    print(f"  Saved 10 images to: {component_dir}")
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# Also save the original images for comparison
original_dir = os.path.join(output_dir, 'original')
os.makedirs(original_dir, exist_ok=True)

for i, (img, label) in enumerate(zip(selected_images, selected_labels)):
    plt.figure(figsize=(2, 2))
    plt.imshow(img.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    
    filename = f'digit_{label}_idx_{i}.png'
    filepath = os.path.join(original_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

print(f"\nOriginal images saved to: {original_dir}")
print(f"\nAll images saved as PNG files in: {output_dir}")
print("Ready for collage creation!")
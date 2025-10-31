#!/usr/bin/env python3
"""
PCA Key Loader Demo
Shows how to reconstruct images from saved PCA "key" (compressed representation)
Demonstrates the compression → decompression process
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def load_and_reconstruct(component_count):
    """Load PCA key and reconstruct images to demonstrate compression/decompression"""
    
    base_dir = 'images/pca_reconstructed_images'
    key_file = os.path.join(base_dir, f'{component_count}_components', 'pca_key.pkl')
    
    if not os.path.exists(key_file):
        print(f"ERROR: PCA key not found at {key_file}")
        return None
    
    # Load the PCA "key"
    with open(key_file, 'rb') as f:
        pca_data = pickle.load(f)
    
    print(f"\n=== PCA Key for {component_count} Components ===")
    print(f"Compression ratio: {pca_data['compression_ratio']:.1f}x")
    print(f"Explained variance: {pca_data['explained_variance']:.3f}")
    print(f"Original image size: 784 pixels (28x28)")
    print(f"Compressed size: {component_count} numbers")
    print(f"Storage saved: {((784 - component_count) / 784 * 100):.1f}%")
    
    # Extract the data
    pca_model = pca_data['pca_model']
    scaler = pca_data['scaler']
    encoded_vectors = pca_data['encoded_vectors']
    original_images = pca_data['original_images']
    labels = pca_data['labels']
    
    print(f"\nEncoded vectors shape: {encoded_vectors.shape}")
    print(f"Each image compressed to {component_count} numbers instead of 784")
    
    # Reconstruct images from compressed representation
    reconstructed_scaled = pca_model.inverse_transform(encoded_vectors)
    reconstructed = scaler.inverse_transform(reconstructed_scaled)
    reconstructed = np.clip(reconstructed, 0, 1)
    
    # Show a comparison for first 5 digits
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i in range(5):
        # Original
        axes[0, i].imshow(original_images[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original\nDigit {labels[i]}', fontsize=12)
        axes[0, i].axis('off')
        
        # Reconstructed
        axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Reconstructed\n{component_count} components', fontsize=12)
        axes[1, i].axis('off')
    
    # Add text showing compression info
    fig.suptitle(f'PCA Compression Demo: 784 → {component_count} numbers ({pca_data["compression_ratio"]:.1f}x compression)', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the comparison
    output_file = os.path.join(base_dir, f'compression_demo_{component_count}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Compression demo saved: {output_file}")
    plt.show()
    
    return pca_data

def show_compression_details():
    """Show detailed breakdown of what gets saved in the PCA key"""
    
    component_count = 50  # Use 50 components as example
    base_dir = 'images/pca_reconstructed_images'
    key_file = os.path.join(base_dir, f'{component_count}_components', 'pca_key.pkl')
    
    if not os.path.exists(key_file):
        print(f"ERROR: Run pca_encode_images.py first to create the PCA key files!")
        return
    
    with open(key_file, 'rb') as f:
        pca_data = pickle.load(f)
    
    print("\n" + "="*60)
    print("WHAT'S IN THE PCA 'KEY'?")
    print("="*60)
    
    pca_model = pca_data['pca_model']
    encoded_vectors = pca_data['encoded_vectors']
    
    print(f"1. PCA Components (the 'dictionary'): {pca_model.components_.shape}")
    print(f"   → {pca_model.components_.shape[0]} learned patterns, each 784 pixels")
    print(f"   → These are the 'building blocks' for reconstruction")
    
    print(f"\n2. Mean image: {pca_model.mean_.shape}")
    print(f"   → The 'average' digit across all training images")
    
    print(f"\n3. Encoded vectors (the 'compressed data'): {encoded_vectors.shape}")
    print(f"   → Each of 10 images compressed to {component_count} numbers")
    print(f"   → Original: 10 × 784 = 7,840 numbers")
    print(f"   → Compressed: 10 × {component_count} = {10 * component_count} numbers")
    print(f"   → Plus shared components: {component_count} × 784 = {component_count * 784} numbers")
    
    total_compressed = (10 * component_count) + (component_count * 784) + 784  # +784 for mean
    total_original = 10 * 784
    
    print(f"\n4. COMPRESSION ANALYSIS:")
    print(f"   → Per image: 784 → {component_count} numbers ({784/component_count:.1f}x smaller)")
    print(f"   → But need to store shared PCA model once")
    print(f"   → For 10 images: Effective compression ≈ {total_original/total_compressed:.2f}x")
    print(f"   → For 1000+ images: Approaches {784/component_count:.1f}x compression!")

# Main execution
if __name__ == "__main__":
    print("PCA Key Loading Demo")
    print("This shows how images are reconstructed from compressed PCA representation")
    
    # Test different component counts
    for components in [10, 50, 100]:
        pca_data = load_and_reconstruct(components)
        if pca_data is None:
            print("Run pca_encode_images.py first!")
            break
    
    # Show detailed breakdown
    show_compression_details()
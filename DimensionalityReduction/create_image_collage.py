#!/usr/bin/env python3
"""
PCA Image Collage Creator
Loads PCA-reconstructed images and creates comparison collages
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob

# Directory containing the reconstructed images
base_dir = 'images/pca_reconstructed_images'

# Check what component folders exist
component_dirs = []
for item in os.listdir(base_dir):
    if 'components' in item and os.path.isdir(os.path.join(base_dir, item)):
        component_dirs.append(item)

component_dirs.sort(key=lambda x: int(x.split('_')[0]))  # Sort by component count
print(f"Found component directories: {component_dirs}")

# Load original images
original_dir = os.path.join(base_dir, 'original')
original_files = sorted(glob.glob(os.path.join(original_dir, '*.png')))

def load_image(filepath):
    """Load PNG image and convert to numpy array"""
    img = Image.open(filepath).convert('L')  # Convert to grayscale
    return np.array(img) / 255.0  # Normalize to [0, 1]

# Create comparison collage
def create_comparison_collage():
    """Create a collage comparing original vs different PCA reconstructions"""
    
    # Set up the figure
    n_rows = len(component_dirs) + 1  # +1 for original row
    n_cols = 10  # 10 digits (0-9)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2.5 * n_rows))
    
    # Load and display original images (top row)
    for col in range(n_cols):
        if col < len(original_files):
            img = load_image(original_files[col])
            axes[0, col].imshow(img, cmap='gray', vmin=0, vmax=1)
            
            # Get digit label from filename
            filename = os.path.basename(original_files[col])
            digit = filename.split('_')[1]
            if col == 0:
                axes[0, col].set_ylabel('Original', rotation=0, labelpad=50, va='center', fontsize=12, fontweight='bold')
            axes[0, col].set_title(f'Digit {digit}', fontsize=10)
        
        axes[0, col].axis('off')
    
    # Load and display reconstructed images
    for row, comp_dir in enumerate(component_dirs, 1):
        comp_path = os.path.join(base_dir, comp_dir)
        comp_files = sorted(glob.glob(os.path.join(comp_path, '*.png')))
        
        # Extract component count for label
        comp_count = comp_dir.split('_')[0]
        
        for col in range(n_cols):
            if col < len(comp_files):
                img = load_image(comp_files[col])
                axes[row, col].imshow(img, cmap='gray', vmin=0, vmax=1)
            
            if col == 0:
                axes[row, col].set_ylabel(f'{comp_count}\nComponents', rotation=0, labelpad=50, va='center', fontsize=12, fontweight='bold')
            
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save the collage
    collage_path = os.path.join(base_dir, 'pca_reconstruction_collage.png')
    plt.savefig(collage_path, dpi=150, bbox_inches='tight')
    print(f"Collage saved to: {collage_path}")
    plt.show()

def create_single_digit_progression():
    """Create a focused comparison for a single digit across different component counts"""
    
    # Use digit 8 (usually interesting for reconstruction)
    digit_to_track = '8'
    
    fig, axes = plt.subplots(1, len(component_dirs) + 1, figsize=(3 * (len(component_dirs) + 1), 3))
    
    # Find and display original
    original_file = None
    for file in original_files:
        if f'digit_{digit_to_track}_' in file:
            original_file = file
            break
    
    if original_file:
        img = load_image(original_file)
        axes[0].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Original', fontsize=14, fontweight='bold')
        axes[0].axis('off')
    
    # Display reconstructions
    for idx, comp_dir in enumerate(component_dirs, 1):
        comp_path = os.path.join(base_dir, comp_dir)
        comp_files = glob.glob(os.path.join(comp_path, f'*digit_{digit_to_track}_*.png'))
        
        if comp_files:
            img = load_image(comp_files[0])
            axes[idx].imshow(img, cmap='gray', vmin=0, vmax=1)
            
            comp_count = comp_dir.split('_')[0]
            axes[idx].set_title(f'{comp_count} Components', fontsize=14, fontweight='bold')
            axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save the progression
    progression_path = os.path.join(base_dir, f'digit_{digit_to_track}_pca_progression.png')
    plt.savefig(progression_path, dpi=150, bbox_inches='tight')
    print(f"Digit {digit_to_track} progression saved to: {progression_path}")
    plt.show()

# Main execution
if __name__ == "__main__":
    print("Creating PCA reconstruction collages...")
    
    if not os.path.exists(original_dir):
        print("ERROR: Original images not found. Run pca_encode_images.py first!")
    else:
        print(f"Found {len(original_files)} original images")
        
        # Create full comparison collage
        create_comparison_collage()
        
        # Create single digit progression
        create_single_digit_progression()
        
        print("\nCollage creation complete!")
        print("Generated files:")
        print("  - pca_reconstruction_collage.png (full comparison)")
        print("  - digit_8_pca_progression.png (single digit focus)")
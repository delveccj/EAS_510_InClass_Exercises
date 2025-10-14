"""
Configuration and setup for SVM examples.
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from packaging import version
import sklearn

def setup_environment():
    """Setup the environment for SVM examples."""
    # Version checks
    assert sys.version_info >= (3, 7), "Python 3.7+ required"
    assert version.parse(sklearn.__version__) >= version.parse("1.0.1"), "scikit-learn 1.0.1+ required"
    
    # Matplotlib configuration
    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14, titlesize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    
    # Create images directory
    images_path = Path() / "images" / "svm"
    images_path.mkdir(parents=True, exist_ok=True)
    
    return images_path

def save_fig(fig_id, images_path=None, tight_layout=True, fig_extension="png", resolution=300):
    """Save matplotlib figure to file."""
    if images_path is None:
        images_path = Path() / "images" / "svm"
    
    path = images_path / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    plt.close()  # Close the figure to free memory
    print(f"Figure saved to: {path}")

# Global setup when module is imported
IMAGES_PATH = setup_environment()
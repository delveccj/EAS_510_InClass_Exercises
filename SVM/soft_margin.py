"""
Soft Margin SVM Examples - Converted from notebook cells.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
try:
    from .config import save_fig, IMAGES_PATH
except ImportError:
    from config import save_fig, IMAGES_PATH

class SoftMarginSVMExamples:
    """Class containing soft margin SVM examples."""
    
    def __init__(self):
        """Initialize with Iris dataset."""
        self.setup_data()
    
    def setup_data(self):
        """Setup the Iris dataset for binary classification."""
        iris = datasets.load_iris(as_frame=True)
        X = iris.data[["petal length (cm)", "petal width (cm)"]].values
        y = iris.target
        
        # Use only setosa (0) and versicolor (1) for binary classification
        setosa_or_versicolor = (y == 0) | (y == 1)
        self.X = X[setosa_or_versicolor]
        self.y = y[setosa_or_versicolor]
        
        print(f"Dataset shape: {self.X.shape}")
        print(f"Classes: {np.unique(self.y)}")
    
    def feature_scaling_comparison(self):
        """Compare SVM performance with and without feature scaling."""
        print("Comparing feature scaling effects...")
        
        # Create dataset with very different scales
        Xs = np.array([[1, 50], [5, 20], [3, 80], [5, 60]]).astype(np.float64)
        ys = np.array([0, 0, 1, 1])
        
        # Train without scaling
        svm_clf = SVC(kernel="linear", C=100)
        svm_clf.fit(Xs, ys)
        
        # Train with scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(Xs)
        svm_clf_scaled = SVC(kernel="linear", C=100)
        svm_clf_scaled.fit(X_scaled, ys)
        
        print(f"Unscaled SVM accuracy: {svm_clf.score(Xs, ys):.3f}")
        print(f"Scaled SVM accuracy: {svm_clf_scaled.score(X_scaled, ys):.3f}")
        
        # Show how scaling affects decision boundary
        self._plot_scaling_comparison(Xs, ys, svm_clf, X_scaled, svm_clf_scaled, scaler)
        
        return svm_clf, svm_clf_scaled
    
    def _plot_scaling_comparison(self, Xs, ys, svm_clf, X_scaled, svm_clf_scaled, scaler):
        """Plot comparison of scaled vs unscaled SVMs."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot unscaled
        axes[0].scatter(Xs[:, 0], Xs[:, 1], c=ys, cmap=plt.cm.RdYlBu, s=100)
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
        axes[0].set_title('Unscaled Features')
        axes[0].grid(True)
        
        # Plot scaled
        axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=ys, cmap=plt.cm.RdYlBu, s=100)
        axes[1].set_xlabel('Scaled Feature 1')
        axes[1].set_ylabel('Scaled Feature 2')
        axes[1].set_title('Scaled Features')
        axes[1].grid(True)
        
        plt.tight_layout()
        save_fig("scaling_comparison", IMAGES_PATH)
        # plt.show() # Disabled for headless mode
    
    def outlier_sensitivity(self):
        """Demonstrate SVM sensitivity to outliers with different C values."""
        print("Testing outlier sensitivity with different C values...")
        
        # Add outliers to the dataset
        X_outliers = np.array([[3.4, 1.3], [3.2, 0.8]])
        y_outliers = np.array([0, 0])
        
        # Dataset with one outlier
        Xo1 = np.concatenate([self.X, X_outliers[:1]], axis=0)
        yo1 = np.concatenate([self.y, y_outliers[:1]], axis=0)
        
        # Dataset with two outliers
        Xo2 = np.concatenate([self.X, X_outliers], axis=0)
        yo2 = np.concatenate([self.y, y_outliers], axis=0)
        
        # Test different C values
        C_values = [0.1, 1, 100]
        datasets = [("Original", self.X, self.y), 
                   ("1 Outlier", Xo1, yo1), 
                   ("2 Outliers", Xo2, yo2)]
        
        for C in C_values:
            print(f"\\nTesting C={C}:")
            for name, X, y in datasets:
                svm_clf = SVC(kernel="linear", C=C)
                svm_clf.fit(X, y)
                accuracy = svm_clf.score(X, y)
                n_support = len(svm_clf.support_)
                print(f"  {name}: accuracy={accuracy:.3f}, support_vectors={n_support}")
        
        # Visualize the effect
        self._plot_outlier_effect(datasets, C_values)
    
    def _plot_outlier_effect(self, datasets, C_values):
        """Plot the effect of outliers on SVM decision boundaries."""
        fig, axes = plt.subplots(len(C_values), len(datasets), figsize=(15, 12))
        
        for i, C in enumerate(C_values):
            for j, (name, X, y) in enumerate(datasets):
                ax = axes[i, j]
                
                # Train SVM
                svm_clf = SVC(kernel="linear", C=C)
                svm_clf.fit(X, y)
                
                # Plot decision boundary
                h = 0.02
                x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
                y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                   np.arange(y_min, y_max, h))
                
                Z = svm_clf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
                scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, s=50)
                
                # Highlight support vectors
                ax.scatter(X[svm_clf.support_, 0], X[svm_clf.support_, 1], 
                          s=100, facecolors='none', edgecolors='black', linewidths=2)
                
                ax.set_title(f'C={C}, {name}')
                ax.set_xlabel('Petal length')
                ax.set_ylabel('Petal width')
        
        plt.tight_layout()
        save_fig("outlier_sensitivity", IMAGES_PATH)
        # plt.show() # Disabled for headless mode
    
    def regularization_comparison(self):
        """Compare different regularization strengths (C values)."""
        print("Comparing regularization strengths...")
        
        # Different C values for comparison
        C_values = [0.1, 1, 10, 100]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, C in enumerate(C_values):
            # Train SVM with current C value
            svm_clf = SVC(kernel="linear", C=C)
            svm_clf.fit(self.X, self.y)
            
            # Plot decision boundary
            ax = axes[i]
            h = 0.02
            x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
            y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))
            
            Z = svm_clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
            scatter = ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap=plt.cm.RdYlBu)
            
            # Highlight support vectors
            ax.scatter(self.X[svm_clf.support_, 0], self.X[svm_clf.support_, 1], 
                      s=100, facecolors='none', edgecolors='black', linewidths=2)
            
            accuracy = svm_clf.score(self.X, self.y)
            n_support = len(svm_clf.support_)
            ax.set_title(f'C={C} (acc={accuracy:.2f}, sv={n_support})')
            ax.set_xlabel('Petal length (cm)')
            ax.set_ylabel('Petal width (cm)')
        
        plt.tight_layout()
        save_fig("regularization_comparison", IMAGES_PATH)
        # plt.show() # Disabled for headless mode
        
        # Print summary
        print("\\nRegularization Effects:")
        for C in C_values:
            svm_clf = SVC(kernel="linear", C=C)
            svm_clf.fit(self.X, self.y)
            print(f"C={C:>5}: accuracy={svm_clf.score(self.X, self.y):.3f}, "
                  f"support_vectors={len(svm_clf.support_)}")
    
    def run_all(self):
        """Run all soft margin SVM examples."""
        print("🎛️ Running Soft Margin SVM Examples")
        print("-" * 40)
        
        self.feature_scaling_comparison()
        print()
        
        self.outlier_sensitivity() 
        print()
        
        self.regularization_comparison()
        print("✅ Soft margin SVM examples completed!")

# For standalone execution
if __name__ == "__main__":
    examples = SoftMarginSVMExamples()
    examples.run_all()
"""
Nonlinear SVM Examples - Converted from notebook cells.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
try:
    from .config import save_fig, IMAGES_PATH
except ImportError:
    from config import save_fig, IMAGES_PATH

class NonlinearSVMExamples:
    """Class containing nonlinear SVM examples."""
    
    def __init__(self):
        """Initialize with make_moons dataset."""
        self.setup_data()
    
    def setup_data(self):
        """Setup the make_moons dataset for nonlinear classification."""
        self.X, self.y = make_moons(n_samples=100, noise=0.15, random_state=42)
        print(f"Dataset shape: {self.X.shape}")
        print(f"Classes: {np.unique(self.y)}")
    
    def polynomial_features_svm(self):
        """Demonstrate polynomial feature transformation with LinearSVC."""
        print("Testing polynomial feature transformation...")
        
        # Create polynomial SVM using feature transformation
        polynomial_svm_clf = make_pipeline(
            PolynomialFeatures(degree=3),
            StandardScaler(),
            LinearSVC(C=10, max_iter=10_000, dual=True, random_state=42)
        )
        polynomial_svm_clf.fit(self.X, self.y)
        
        accuracy = polynomial_svm_clf.score(self.X, self.y)
        print(f"Polynomial SVM accuracy: {accuracy:.3f}")
        
        # Visualize
        self._plot_decision_boundary(polynomial_svm_clf, "Polynomial Features SVM")
        
        return polynomial_svm_clf
    
    def polynomial_kernel_svm(self):
        """Demonstrate polynomial kernel SVM."""
        print("Testing polynomial kernel SVM...")
        
        # Different polynomial kernels
        kernels = [
            ("Poly degree=3", SVC(kernel="poly", degree=3, coef0=1, C=5)),
            ("Poly degree=10", SVC(kernel="poly", degree=10, coef0=100, C=5))
        ]
        
        results = {}
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, (name, svm_clf) in enumerate(kernels):
            # Create pipeline with scaling
            poly_kernel_svm_clf = make_pipeline(StandardScaler(), svm_clf)
            poly_kernel_svm_clf.fit(self.X, self.y)
            
            accuracy = poly_kernel_svm_clf.score(self.X, self.y)
            print(f"{name} accuracy: {accuracy:.3f}")
            results[name] = poly_kernel_svm_clf
            
            # Plot decision boundary
            ax = axes[i]
            self._plot_predictions(poly_kernel_svm_clf, ax, [-1.5, 2.45, -1, 1.5])
            self._plot_dataset(self.X, self.y, ax, [-1.5, 2.4, -1, 1.5])
            ax.set_title(f"{name} (acc={accuracy:.2f})")
        
        plt.tight_layout()
        save_fig("polynomial_kernels", IMAGES_PATH)
        # plt.show() # Disabled for headless mode
        
        return results
    
    def rbf_kernel_svm(self):
        """Demonstrate RBF kernel SVM."""
        print("Testing RBF kernel SVM...")
        
        # Simple RBF kernel
        rbf_kernel_svm_clf = make_pipeline(StandardScaler(),
                                         SVC(kernel="rbf", gamma=5, C=0.001))
        rbf_kernel_svm_clf.fit(self.X, self.y)
        
        accuracy = rbf_kernel_svm_clf.score(self.X, self.y)
        print(f"RBF SVM accuracy: {accuracy:.3f}")
        
        # Visualize
        self._plot_decision_boundary(rbf_kernel_svm_clf, "RBF Kernel SVM")
        
        return rbf_kernel_svm_clf
    
    def rbf_hyperparameter_comparison(self):
        """Compare different RBF hyperparameters."""
        print("Comparing RBF hyperparameters...")
        
        # Different gamma and C combinations
        gamma1, gamma2 = 0.1, 5
        C1, C2 = 0.001, 1000
        hyperparams = [(gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        svm_clfs = []
        for i, (gamma, C) in enumerate(hyperparams):
            # Create and train RBF SVM
            rbf_kernel_svm_clf = make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", gamma=gamma, C=C)
            )
            rbf_kernel_svm_clf.fit(self.X, self.y)
            svm_clfs.append(rbf_kernel_svm_clf)
            
            accuracy = rbf_kernel_svm_clf.score(self.X, self.y)
            print(f"γ={gamma}, C={C}: accuracy={accuracy:.3f}")
            
            # Plot decision boundary
            ax = axes[i]
            self._plot_predictions(rbf_kernel_svm_clf, ax, [-1.5, 2.45, -1, 1.5])
            self._plot_dataset(self.X, self.y, ax, [-1.5, 2.45, -1, 1.5])
            ax.set_title(f"γ={gamma}, C={C} (acc={accuracy:.2f})")
        
        plt.tight_layout()
        save_fig("rbf_hyperparameters", IMAGES_PATH)
        # plt.show() # Disabled for headless mode
        
        return svm_clfs
    
    def gaussian_rbf_demonstration(self):
        """Demonstrate how Gaussian RBF works."""
        print("Demonstrating Gaussian RBF transformation...")
        
        def gaussian_rbf(x, landmark, gamma):
            return np.exp(-gamma * np.linalg.norm(x - landmark, axis=1)**2)
        
        # Create 1D dataset for demonstration
        X1D = np.linspace(-4.5, 4.5, 200).reshape(-1, 1)
        gamma = 0.3
        
        # Calculate RBF features with two landmarks
        x2s = gaussian_rbf(X1D, -2, gamma)
        x3s = gaussian_rbf(X1D, 1, gamma)
        
        # Plot the RBF transformation
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Original 1D space
        axes[0].plot(X1D, np.zeros_like(X1D), 'b.')
        axes[0].axvline(x=-2, color='r', linestyle='--', label='Landmark 1')
        axes[0].axvline(x=1, color='g', linestyle='--', label='Landmark 2')
        axes[0].set_xlabel('x')
        axes[0].set_title('Original 1D Space')
        axes[0].legend()
        axes[0].grid(True)
        
        # RBF features
        axes[1].plot(X1D.ravel(), x2s, 'r-', label='RBF(-2)')
        axes[1].plot(X1D.ravel(), x3s, 'g-', label='RBF(1)')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('RBF value')
        axes[1].set_title('RBF Features')
        axes[1].legend()
        axes[1].grid(True)
        
        # 2D transformed space
        axes[2].plot(x2s, x3s, 'b-')
        axes[2].set_xlabel('RBF(-2)')
        axes[2].set_ylabel('RBF(1)')
        axes[2].set_title('2D Transformed Space')
        axes[2].grid(True)
        
        plt.tight_layout()
        save_fig("gaussian_rbf_demo", IMAGES_PATH)
        # plt.show() # Disabled for headless mode
        
        print("RBF transformation maps data to higher dimensional space where it becomes linearly separable!")
    
    def _plot_dataset(self, X, y, ax, axes):
        """Plot the dataset points."""
        ax.plot(X[:, 0][y==0], X[:, 1][y==0], "bs", label="Class 0")
        ax.plot(X[:, 0][y==1], X[:, 1][y==1], "g^", label="Class 1")
        ax.axis(axes)
        ax.grid(True)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$", rotation=0)
    
    def _plot_predictions(self, clf, ax, axes):
        """Plot SVM decision boundary."""
        x0s = np.linspace(axes[0], axes[1], 100)
        x1s = np.linspace(axes[2], axes[3], 100)
        x0, x1 = np.meshgrid(x0s, x1s)
        X_grid = np.c_[x0.ravel(), x1.ravel()]
        y_pred = clf.predict(X_grid).reshape(x0.shape)
        
        ax.contourf(x0, x1, y_pred, alpha=0.3, cmap=plt.cm.RdYlBu)
        ax.contour(x0, x1, y_pred, colors='black', linestyles='--', linewidths=0.5)
    
    def _plot_decision_boundary(self, clf, title):
        """Plot decision boundary for a classifier."""
        plt.figure(figsize=(10, 6))
        
        # Create mesh
        h = 0.02
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # Plot decision boundary
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
        scatter = plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap=plt.cm.RdYlBu)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title(title)
        plt.colorbar(scatter)
        
        # Save figure
        save_fig(title.lower().replace(' ', '_'), IMAGES_PATH)
        # plt.show() # Disabled for headless mode
    
    def compare_kernels(self):
        """Compare different kernel types side by side."""
        print("Comparing different kernel types...")
        
        kernels = [
            ("Linear", SVC(kernel="linear", C=1)),
            ("Polynomial", SVC(kernel="poly", degree=3, coef0=1, C=5)),
            ("RBF", SVC(kernel="rbf", gamma=5, C=0.001)),
            ("Sigmoid", SVC(kernel="sigmoid", gamma=0.1, coef0=1, C=1))
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, (name, svm_clf) in enumerate(kernels):
            # Create pipeline with scaling
            kernel_svm_clf = make_pipeline(StandardScaler(), svm_clf)
            kernel_svm_clf.fit(self.X, self.y)
            
            accuracy = kernel_svm_clf.score(self.X, self.y)
            print(f"{name} kernel accuracy: {accuracy:.3f}")
            
            # Plot decision boundary
            ax = axes[i]
            self._plot_predictions(kernel_svm_clf, ax, [-1.5, 2.45, -1, 1.5])
            self._plot_dataset(self.X, self.y, ax, [-1.5, 2.45, -1, 1.5])
            ax.set_title(f"{name} Kernel (acc={accuracy:.2f})")
            if i == 0:
                ax.legend()
        
        plt.tight_layout()
        save_fig("kernel_comparison", IMAGES_PATH)
        # plt.show() # Disabled for headless mode
    
    def run_all(self):
        """Run all nonlinear SVM examples."""
        print("🎪 Running Nonlinear SVM Examples")
        print("-" * 40)
        
        self.polynomial_features_svm()
        print()
        
        self.polynomial_kernel_svm()
        print()
        
        self.rbf_kernel_svm()
        print()
        
        self.rbf_hyperparameter_comparison()
        print()
        
        self.gaussian_rbf_demonstration()
        print()
        
        self.compare_kernels()
        print("✅ Nonlinear SVM examples completed!")

# For standalone execution
if __name__ == "__main__":
    examples = NonlinearSVMExamples()
    examples.run_all()
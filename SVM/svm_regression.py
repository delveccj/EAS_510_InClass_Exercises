"""
SVM Regression Examples - Converted from notebook cells.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
try:
    from .config import save_fig, IMAGES_PATH
except ImportError:
    from config import save_fig, IMAGES_PATH

class SVMRegressionExamples:
    """Class containing SVM regression examples."""
    
    def __init__(self):
        """Initialize with synthetic quadratic dataset."""
        self.setup_data()
    
    def setup_data(self):
        """Setup synthetic quadratic dataset for regression."""
        np.random.seed(42)
        self.X = 2 * np.random.rand(50, 1) - 1
        self.y = 0.2 + 0.1 * self.X[:, 0] + 0.5 * self.X[:, 0] ** 2 + np.random.randn(50) / 10
        
        print(f"Dataset shape: {self.X.shape}")
        print(f"Target range: [{self.y.min():.2f}, {self.y.max():.2f}]")
    
    def polynomial_svr(self):
        """Demonstrate polynomial SVM regression."""
        print("Testing polynomial SVM regression...")
        
        # Different C values for comparison
        svm_poly_reg1 = make_pipeline(StandardScaler(),
                                     SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1))
        svm_poly_reg2 = make_pipeline(StandardScaler(),
                                     SVR(kernel="poly", degree=2, C=100, epsilon=0.1))
        
        # Fit models
        svm_poly_reg1.fit(self.X, self.y)
        svm_poly_reg2.fit(self.X, self.y)
        
        # Calculate scores
        score1 = svm_poly_reg1.score(self.X, self.y)
        score2 = svm_poly_reg2.score(self.X, self.y)
        
        print(f"Low C (C=0.01) R² score: {score1:.3f}")
        print(f"High C (C=100) R² score: {score2:.3f}")
        
        # Visualize
        self._plot_svr_comparison([svm_poly_reg1, svm_poly_reg2], 
                                 ["Low C (C=0.01)", "High C (C=100)"],
                                 "Polynomial SVR Comparison")
        
        return svm_poly_reg1, svm_poly_reg2
    
    def rbf_svr(self):
        """Demonstrate RBF SVM regression."""
        print("Testing RBF SVM regression...")
        
        # Different gamma and C combinations
        svm_rbf_reg1 = make_pipeline(StandardScaler(),
                                    SVR(kernel="rbf", gamma=0.1, C=1, epsilon=0.1))
        svm_rbf_reg2 = make_pipeline(StandardScaler(),
                                    SVR(kernel="rbf", gamma=1, C=1, epsilon=0.1))
        
        # Fit models
        svm_rbf_reg1.fit(self.X, self.y)
        svm_rbf_reg2.fit(self.X, self.y)
        
        # Calculate scores
        score1 = svm_rbf_reg1.score(self.X, self.y)
        score2 = svm_rbf_reg2.score(self.X, self.y)
        
        print(f"Low γ (γ=0.1) R² score: {score1:.3f}")
        print(f"High γ (γ=1.0) R² score: {score2:.3f}")
        
        # Visualize
        self._plot_svr_comparison([svm_rbf_reg1, svm_rbf_reg2], 
                                 ["Low γ (γ=0.1)", "High γ (γ=1.0)"],
                                 "RBF SVR Comparison")
        
        return svm_rbf_reg1, svm_rbf_reg2
    
    def epsilon_sensitivity(self):
        """Demonstrate effect of epsilon parameter."""
        print("Testing epsilon sensitivity...")
        
        # Different epsilon values
        epsilons = [0.01, 0.1, 0.5]
        models = []
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, eps in enumerate(epsilons):
            # Create SVR with different epsilon
            svm_reg = make_pipeline(StandardScaler(),
                                  SVR(kernel="rbf", gamma=1, C=1, epsilon=eps))
            svm_reg.fit(self.X, self.y)
            models.append(svm_reg)
            
            score = svm_reg.score(self.X, self.y)
            print(f"ε={eps}: R² score={score:.3f}")
            
            # Plot
            ax = axes[i]
            self._plot_single_svr(svm_reg, ax, f"ε={eps} (R²={score:.2f})")
        
        plt.tight_layout()
        save_fig("epsilon_sensitivity", IMAGES_PATH)
        # plt.show() # Disabled for headless mode
        
        return models
    
    def _plot_svr_comparison(self, models, labels, title):
        """Plot comparison of SVR models."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, (model, label) in enumerate(zip(models, labels)):
            ax = axes[i]
            self._plot_single_svr(model, ax, label)
        
        plt.suptitle(title)
        plt.tight_layout()
        save_fig(title.lower().replace(' ', '_'), IMAGES_PATH)
        # plt.show() # Disabled for headless mode
    
    def _plot_single_svr(self, model, ax, title):
        """Plot single SVR model."""
        # Create prediction line
        X_plot = np.linspace(-1, 1, 100).reshape(-1, 1)
        y_pred = model.predict(X_plot)
        
        # Plot data points
        ax.scatter(self.X, self.y, color='blue', alpha=0.6, label='Data')
        
        # Plot regression line
        ax.plot(X_plot, y_pred, color='red', linewidth=2, label='SVR prediction')
        
        # Add epsilon tube (approximation)
        epsilon = model.named_steps['svr'].epsilon
        ax.fill_between(X_plot.ravel(), y_pred - epsilon, y_pred + epsilon, 
                       alpha=0.2, color='gray', label=f'ε-tube (ε={epsilon})')
        
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
    
    def compare_kernels(self):
        """Compare different kernels for regression."""
        print("Comparing different kernels for regression...")
        
        kernels = [
            ("Linear", SVR(kernel="linear", C=1, epsilon=0.1)),
            ("Polynomial", SVR(kernel="poly", degree=2, C=1, epsilon=0.1)),
            ("RBF", SVR(kernel="rbf", gamma=1, C=1, epsilon=0.1)),
            ("Sigmoid", SVR(kernel="sigmoid", gamma=0.1, C=1, epsilon=0.1))
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, (name, svr) in enumerate(kernels):
            # Create pipeline with scaling
            model = make_pipeline(StandardScaler(), svr)
            model.fit(self.X, self.y)
            
            score = model.score(self.X, self.y)
            print(f"{name} kernel R² score: {score:.3f}")
            
            # Plot
            ax = axes[i]
            self._plot_single_svr(model, ax, f"{name} (R²={score:.2f})")
        
        plt.tight_layout()
        save_fig("svr_kernel_comparison", IMAGES_PATH)
        # plt.show() # Disabled for headless mode
    
    def run_all(self):
        """Run all SVM regression examples."""
        print("📊 Running SVM Regression Examples")
        print("-" * 40)
        
        self.polynomial_svr()
        print()
        
        self.rbf_svr()
        print()
        
        self.epsilon_sensitivity()
        print()
        
        self.compare_kernels()
        print("✅ SVM regression examples completed!")

# For standalone execution
if __name__ == "__main__":
    examples = SVMRegressionExamples()
    examples.run_all()
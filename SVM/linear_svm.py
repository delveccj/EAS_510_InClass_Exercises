"""
Linear SVM Examples - Converted from notebook cells.
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

class LinearSVMExamples:
    """Class containing linear SVM examples."""
    
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
    
    def basic_linear_svm(self):
        """Basic Linear SVM example."""
        print("Running basic Linear SVM...")
        
        # Create and train SVM
        svm_clf = SVC(kernel="linear", C=1)
        svm_clf.fit(self.X, self.y)
        
        # Test predictions
        X_new = [[5.5, 1.7], [5.0, 1.5]]
        predictions = svm_clf.predict(X_new)
        decision_scores = svm_clf.decision_function(X_new)
        
        print(f"Predictions: {predictions}")
        print(f"Decision scores: {decision_scores}")
        
        return svm_clf
    
    def scaled_linear_svm(self):
        """Linear SVM with feature scaling."""
        print("Running scaled Linear SVM comparison...")
        
        scaler = StandardScaler()
        svm_clf1 = LinearSVC(C=1, max_iter=10_000, dual=True, random_state=42)
        svm_clf2 = LinearSVC(C=100, max_iter=10_000, dual=True, random_state=42)
        
        # Create pipelines with scaling
        scaled_svm_clf1 = make_pipeline(scaler, svm_clf1)
        scaled_svm_clf2 = make_pipeline(scaler, svm_clf2)
        
        # Fit models
        scaled_svm_clf1.fit(self.X, self.y)
        scaled_svm_clf2.fit(self.X, self.y)
        
        print(f"Model 1 accuracy: {scaled_svm_clf1.score(self.X, self.y):.3f}")
        print(f"Model 2 accuracy: {scaled_svm_clf2.score(self.X, self.y):.3f}")
        
        return scaled_svm_clf1, scaled_svm_clf2
    
    def visualize_decision_boundary(self):
        """Visualize SVM decision boundary."""
        print("Creating decision boundary visualization...")
        
        # Simple SVM
        svm_clf = SVC(kernel="linear", C=1)
        svm_clf.fit(self.X, self.y)
        
        # Create a mesh for plotting
        h = 0.02
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Plot decision boundary
        plt.figure(figsize=(10, 6))
        Z = svm_clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
        scatter = plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap=plt.cm.RdYlBu)
        plt.xlabel('Petal length (cm)')
        plt.ylabel('Petal width (cm)')
        plt.title('Linear SVM Decision Boundary')
        plt.colorbar(scatter)
        
        # Save figure
        save_fig("linear_svm_decision_boundary", IMAGES_PATH)
    
    def run_all(self):
        """Run all linear SVM examples."""
        print("🎯 Running Linear SVM Examples")
        print("-" * 40)
        
        self.basic_linear_svm()
        print()
        
        self.scaled_linear_svm()
        print()
        
        self.visualize_decision_boundary()
        print("✅ Linear SVM examples completed!")

# For standalone execution
if __name__ == "__main__":
    examples = LinearSVMExamples()
    examples.run_all()
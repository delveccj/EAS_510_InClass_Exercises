from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_moons

# Create images directory if it doesn't exist
os.makedirs('images/ensemble', exist_ok=True)



X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bootstrap Aggregating (Bagging) Classifier
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=100,
    n_jobs=-1,
    random_state=42,
    bootstrap=True
)

# Train the model
bag_clf.fit(X_train, y_train)

# Make predictions
y_pred = bag_clf.predict(X_test)

# Summary Statistics
print("=== BAGGING CLASSIFIER RESULTS ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Number of estimators: {bag_clf.n_estimators}")
print(f"Max samples per estimator: {bag_clf.max_samples}")

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== DETAILED CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# Feature importance (averaged across all trees)
print("\n=== TOP FEATURE IMPORTANCES ===")
importances = np.mean([tree.feature_importances_ for tree in bag_clf.estimators_], axis=0)
for i, importance in enumerate(importances):
    print(f"Feature {i}: {importance:.4f}")

# Decision Boundary Visualization
def plot_decision_boundary(clf, X, y, title, filename):
    plt.figure(figsize=(12, 5))

    # Create a mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # Plot decision boundary
    plt.subplot(1, 2, 1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)

    # Plot training points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'{title} - Decision Boundary')
    plt.colorbar(scatter)

    # Plot prediction probabilities
    plt.subplot(1, 2, 2)
    Z_proba = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z_proba = Z_proba.reshape(xx.shape)
    contour = plt.contourf(xx, yy, Z_proba, levels=20, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'{title} - Prediction Probabilities')
    plt.colorbar(contour)

    plt.tight_layout()
    plt.savefig(f'images/ensemble/{filename}', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Decision boundary plot saved to: images/ensemble/{filename}")

# Generate decision boundary plot
plot_decision_boundary(bag_clf, X_train, y_train,
                    "Bagging Classifier (500 Trees)",
                    "bagging_decision_boundary.png")

# Optional: Compare with single decision tree
single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)
plot_decision_boundary(single_tree, X_train, y_train,
                    "Single Decision Tree",
                    "single_tree_decision_boundary.png")

print(f"\nSingle Tree Accuracy: {accuracy_score(y_test, single_tree.predict(X_test)):.4f}")
print(f"Bagging Accuracy: {accuracy_score(y_test, y_pred):.4f}")
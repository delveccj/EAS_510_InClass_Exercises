#!/usr/bin/env python3
"""
Out-of-Bag (OOB) Scoring Demo
=============================

Demonstrates how OOB scoring provides free validation in bagging ensembles
using the classic make_moons dataset.

Key Learning Points:
- OOB score vs test score comparison
- Free validation without separate validation set
- How OOB estimates generalization performance

Author: Course Materials - EAS 510 BAI
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Create output directory
os.makedirs('images/ensemble', exist_ok=True)

# Generate crescent moon data
print("🌙 Generating make_moons dataset...")
X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Dataset Info:")
print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")
print(f"   Features: {X.shape[1]}")

# Train bagging classifier with OOB scoring
print("\n🎒 Training Bagging Classifier with OOB scoring...")
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,
    oob_score=True,  # Enable OOB
    random_state=42
)

bag_clf.fit(X_train, y_train)

# Get predictions and scores
y_pred_test = bag_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
oob_score = bag_clf.oob_score_

print(f"\n📈 Results:")
print(f"   OOB Score: {oob_score:.4f}")
print(f"   Test Accuracy: {test_accuracy:.4f}")
print(f"   Difference: {abs(oob_score - test_accuracy):.4f}")

# Compare with single decision tree
print("\n🌳 Single Decision Tree for comparison...")
single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)
single_tree_accuracy = single_tree.score(X_test, y_test)

print(f"   Single Tree Test Accuracy: {single_tree_accuracy:.4f}")
print(f"   Bagging Improvement: +{test_accuracy - single_tree_accuracy:.4f}")

# Visualize the results
def plot_decision_boundary_comparison():
    """Plot decision boundaries for single tree vs bagging ensemble"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Create mesh for decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Plot 1: Original data
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    axes[0].set_title('🌙 Make Moons Dataset')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    
    # Plot 2: Single decision tree
    Z_single = single_tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_single = Z_single.reshape(xx.shape)
    axes[1].contourf(xx, yy, Z_single, alpha=0.8, cmap=plt.cm.RdYlBu)
    scatter1 = axes[1].scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                              cmap=plt.cm.RdYlBu, edgecolors='black')
    axes[1].set_title(f'🌳 Single Tree\nAccuracy: {single_tree_accuracy:.3f}')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    
    # Plot 3: Bagging ensemble
    Z_bag = bag_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_bag = Z_bag.reshape(xx.shape)
    axes[2].contourf(xx, yy, Z_bag, alpha=0.8, cmap=plt.cm.RdYlBu)
    scatter2 = axes[2].scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                              cmap=plt.cm.RdYlBu, edgecolors='black')
    axes[2].set_title(f'🎒 Bagging Ensemble\nOOB: {oob_score:.3f} | Test: {test_accuracy:.3f}')
    axes[2].set_xlabel('Feature 1')
    axes[2].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig('images/ensemble/oob_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("💾 Decision boundary comparison saved to: images/ensemble/oob_comparison.png")

# Generate visualization
print("\n🎨 Creating decision boundary visualization...")
plot_decision_boundary_comparison()

# Demonstrate OOB sample tracking
print(f"\n🔍 OOB Sample Analysis:")
print(f"   Total training samples: {len(X_train)}")
print(f"   Number of estimators: {bag_clf.n_estimators}")

# Show how many times each sample was used
oob_decision_function = bag_clf.oob_decision_function_
n_oob_predictions = np.sum(~np.isnan(oob_decision_function), axis=0)

print(f"   Samples used for OOB: {len(oob_decision_function)}")
print(f"   Average predictions per sample: {np.mean(n_oob_predictions):.1f}")
print(f"   Min predictions per sample: {np.min(n_oob_predictions)}")
print(f"   Max predictions per sample: {np.max(n_oob_predictions)}")

# Compare OOB vs test accuracy across different ensemble sizes
print(f"\n📊 OOB vs Test Accuracy across ensemble sizes:")
ensemble_sizes = [10, 25, 50, 100, 200, 500]
oob_scores = []
test_scores = []

for n_est in ensemble_sizes:
    # Train with different ensemble sizes
    bag_temp = BaggingClassifier(
        DecisionTreeClassifier(),
        n_estimators=n_est,
        oob_score=True,
        random_state=42
    )
    bag_temp.fit(X_train, y_train)
    
    oob_score_temp = bag_temp.oob_score_
    test_score_temp = bag_temp.score(X_test, y_test)
    
    oob_scores.append(oob_score_temp)
    test_scores.append(test_score_temp)
    
    print(f"   {n_est:3d} trees: OOB={oob_score_temp:.4f}, Test={test_score_temp:.4f}, Diff={abs(oob_score_temp-test_score_temp):.4f}")

# Plot OOB vs Test accuracy
plt.figure(figsize=(10, 6))
plt.plot(ensemble_sizes, oob_scores, 'o-', label='OOB Score', linewidth=2, markersize=8)
plt.plot(ensemble_sizes, test_scores, 's--', label='Test Accuracy', linewidth=2, markersize=8)
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('🎯 OOB Score vs Test Accuracy\n(How well does OOB estimate true performance?)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/ensemble/oob_vs_test_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()
print("💾 OOB vs Test accuracy plot saved to: images/ensemble/oob_vs_test_accuracy.png")

print(f"\n✅ Demo complete!")
print(f"🎒 Key Takeaway: OOB score ({oob_score:.4f}) closely estimates test performance ({test_accuracy:.4f})")
print(f"📈 This gives you FREE validation without a separate validation set!")
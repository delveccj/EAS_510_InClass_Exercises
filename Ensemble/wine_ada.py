"""
AdaBoost vs Decision Tree Comparison on Wine Dataset
EAS 510 - Basics of AI

This script compares the performance of a single Decision Tree 
vs AdaBoost ensemble on the Wine dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load the Wine dataset
print("🍷 Loading Wine Dataset...")
wine = load_wine()
X, y = wine.data, wine.target

print(f"Dataset shape: {X.shape}")
print(f"Features: {wine.feature_names}")
print(f"Classes: {wine.target_names}")
print()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print()

# 1. Single Decision Tree
print("🌳 Training Single Decision Tree...")
dt_classifier = DecisionTreeClassifier(
    max_depth=None,  # Let it grow deep
    random_state=42
)
dt_classifier.fit(X_train, y_train)

# 2. AdaBoost with Decision Tree stumps
print("🚀 Training AdaBoost Ensemble...")
ada_classifier = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=2),  # Stumps!
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada_classifier.fit(X_train, y_train)

print()

# Compare performance
print("📊 Performance Comparison:")
print("=" * 50)

# Cross-validation scores
dt_cv_scores = cross_val_score(dt_classifier, X_train, y_train, cv=5)
ada_cv_scores = cross_val_score(ada_classifier, X_train, y_train, cv=5)

print(f"Decision Tree CV Score: {dt_cv_scores.mean():.3f} (+/- {dt_cv_scores.std() * 2:.3f})")
print(f"AdaBoost CV Score:      {ada_cv_scores.mean():.3f} (+/- {ada_cv_scores.std() * 2:.3f})")
print()

# Test set performance
dt_test_score = dt_classifier.score(X_test, y_test)
ada_test_score = ada_classifier.score(X_test, y_test)

print(f"Decision Tree Test Accuracy: {dt_test_score:.3f}")
print(f"AdaBoost Test Accuracy:      {ada_test_score:.3f}")
print()

if ada_test_score > dt_test_score:
    improvement = (ada_test_score - dt_test_score) * 100
    print(f"🎉 AdaBoost wins by {improvement:.1f} percentage points!")
else:
    print("🤔 Decision Tree performed better this time.")

print()

# Detailed classification report for AdaBoost
print("🔍 Detailed AdaBoost Results:")
print("-" * 30)
ada_predictions = ada_classifier.predict(X_test)
print(classification_report(y_test, ada_predictions, target_names=wine.target_names))

# Show which features the first few stumps chose
print("🔍 What Features Did AdaBoost Choose?")
print("-" * 40)
for i in range(min(5, len(ada_classifier.estimators_))):
    stump = ada_classifier.estimators_[i]
    feature_idx = stump.tree_.feature[0]  # Root node feature
    threshold = stump.tree_.threshold[0]  # Root node threshold
    weight = ada_classifier.estimator_weights_[i]
    
    feature_name = wine.feature_names[feature_idx]
    print(f"Stump {i+1}: {feature_name} > {threshold:.2f} (weight: {weight:.3f})")

print()

# Plot feature importance comparison
plt.figure(figsize=(12, 8))

# Decision Tree feature importance
plt.subplot(2, 1, 1)
dt_importance = dt_classifier.feature_importances_
plt.barh(range(len(wine.feature_names)), dt_importance)
plt.yticks(range(len(wine.feature_names)), wine.feature_names, fontsize=8)
plt.xlabel('Feature Importance')
plt.title('Decision Tree Feature Importance')
plt.grid(True, alpha=0.3)

# AdaBoost feature importance
plt.subplot(2, 1, 2)
ada_importance = ada_classifier.feature_importances_
plt.barh(range(len(wine.feature_names)), ada_importance)
plt.yticks(range(len(wine.feature_names)), wine.feature_names, fontsize=8)
plt.xlabel('Feature Importance')
plt.title('AdaBoost Feature Importance')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('wine_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot confusion matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Decision Tree confusion matrix
dt_pred = dt_classifier.predict(X_test)
dt_cm = confusion_matrix(y_test, dt_pred)
sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=wine.target_names, yticklabels=wine.target_names, ax=ax1)
ax1.set_title('Decision Tree Confusion Matrix')
ax1.set_ylabel('True Label')
ax1.set_xlabel('Predicted Label')

# AdaBoost confusion matrix
ada_cm = confusion_matrix(y_test, ada_predictions)
sns.heatmap(ada_cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=wine.target_names, yticklabels=wine.target_names, ax=ax2)
ax2.set_title('AdaBoost Confusion Matrix')
ax2.set_ylabel('True Label')
ax2.set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('wine_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

print("🎯 Key Takeaways:")
print("1. AdaBoost combines many weak stumps into a strong classifier")
print("2. Each stump specializes on different chemical features")
print("3. The ensemble approach often outperforms a single complex tree")
print("4. Feature importance patterns may differ between approaches")
print("\n📁 Saved plots: wine_feature_importance_comparison.png, wine_confusion_matrices.png")
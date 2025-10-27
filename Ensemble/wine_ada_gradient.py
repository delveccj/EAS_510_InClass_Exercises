"""
AdaBoost vs Gradient Boosting Comparison on Wine Dataset
EAS 510 - Basics of AI

This script compares AdaBoost and Gradient Boosting ensemble methods
on the Wine dataset to demonstrate their different approaches to
sequential learning.

Key Differences:
- AdaBoost: Changes sample weights, focuses on misclassified samples
- Gradient Boosting: Fits residual errors from previous predictions
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

# 1. AdaBoost Classifier
print("🚀 Training AdaBoost Ensemble...")
ada_classifier = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # Decision stumps
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada_classifier.fit(X_train, y_train)

# 2. Gradient Boosting Classifier
print("🌟 Training Gradient Boosting Ensemble...")
gb_classifier = GradientBoostingClassifier(
    max_depth=3,  # Slightly deeper trees than stumps
    n_estimators=50,
    learning_rate=0.1,  # Lower learning rate for GB
    random_state=42
)
gb_classifier.fit(X_train, y_train)

print()

# Compare performance
print("📊 Performance Comparison:")
print("=" * 60)

# Cross-validation scores
ada_cv_scores = cross_val_score(ada_classifier, X_train, y_train, cv=5)
gb_cv_scores = cross_val_score(gb_classifier, X_train, y_train, cv=5)

print(f"AdaBoost CV Score:        {ada_cv_scores.mean():.3f} (+/- {ada_cv_scores.std() * 2:.3f})")
print(f"Gradient Boosting CV Score: {gb_cv_scores.mean():.3f} (+/- {gb_cv_scores.std() * 2:.3f})")
print()

# Test set performance
ada_test_score = ada_classifier.score(X_test, y_test)
gb_test_score = gb_classifier.score(X_test, y_test)

print(f"AdaBoost Test Accuracy:        {ada_test_score:.3f}")
print(f"Gradient Boosting Test Accuracy: {gb_test_score:.3f}")
print()

if gb_test_score > ada_test_score:
    improvement = (gb_test_score - ada_test_score) * 100
    print(f"🎉 Gradient Boosting wins by {improvement:.1f} percentage points!")
elif ada_test_score > gb_test_score:
    improvement = (ada_test_score - gb_test_score) * 100
    print(f"🚀 AdaBoost wins by {improvement:.1f} percentage points!")
else:
    print("🤝 It's a tie!")

print()

# Detailed classification reports
print("🔍 Detailed AdaBoost Results:")
print("-" * 40)
ada_predictions = ada_classifier.predict(X_test)
print(classification_report(y_test, ada_predictions, target_names=wine.target_names))

print("🔍 Detailed Gradient Boosting Results:")
print("-" * 40)
gb_predictions = gb_classifier.predict(X_test)
print(classification_report(y_test, gb_predictions, target_names=wine.target_names))

# Show which features the algorithms prioritize
print("🔍 Feature Selection Analysis:")
print("-" * 40)

print("\nAdaBoost Stumps (first 5):")
for i in range(min(5, len(ada_classifier.estimators_))):
    stump = ada_classifier.estimators_[i]
    feature_idx = stump.tree_.feature[0]  # Root node feature
    threshold = stump.tree_.threshold[0]  # Root node threshold
    weight = ada_classifier.estimator_weights_[i]
    
    feature_name = wine.feature_names[feature_idx]
    print(f"  Stump {i+1}: {feature_name} > {threshold:.2f} (weight: {weight:.3f})")

print(f"\nGradient Boosting Trees (max_depth={gb_classifier.max_depth}):")
print("  Each tree predicts residual errors from previous ensemble")
print(f"  Learning rate: {gb_classifier.learning_rate}")
print(f"  Total trees: {gb_classifier.n_estimators}")

print()

# Training progression analysis
print("📈 Training Progression Analysis:")
print("-" * 40)

# Get staged predictions to see how accuracy improves
ada_staged_scores = []
gb_staged_scores = []

# AdaBoost staged predictions
for pred in ada_classifier.staged_predict(X_test):
    ada_staged_scores.append(accuracy_score(y_test, pred))

# Gradient Boosting staged predictions  
for pred in gb_classifier.staged_predict(X_test):
    gb_staged_scores.append(accuracy_score(y_test, pred))

# Plot training progression
plt.figure(figsize=(15, 10))

# Training progression subplot
plt.subplot(2, 2, 1)
plt.plot(range(1, len(ada_staged_scores) + 1), ada_staged_scores, 
         'b-', label='AdaBoost', linewidth=2)
plt.plot(range(1, len(gb_staged_scores) + 1), gb_staged_scores, 
         'r-', label='Gradient Boosting', linewidth=2)
plt.xlabel('Number of Estimators')
plt.ylabel('Test Accuracy')
plt.title('Training Progression: Accuracy vs Number of Trees')
plt.legend()
plt.grid(True, alpha=0.3)

# Feature importance comparison
plt.subplot(2, 2, 2)
ada_importance = ada_classifier.feature_importances_
gb_importance = gb_classifier.feature_importances_

# Select top 10 most important features for clarity
top_features = np.argsort(ada_importance + gb_importance)[-10:]
feature_names = [wine.feature_names[i] for i in top_features]

x = np.arange(len(top_features))
width = 0.35

plt.barh(x - width/2, ada_importance[top_features], width, 
         label='AdaBoost', alpha=0.8, color='skyblue')
plt.barh(x + width/2, gb_importance[top_features], width, 
         label='Gradient Boosting', alpha=0.8, color='lightcoral')

plt.yticks(x, feature_names, fontsize=8)
plt.xlabel('Feature Importance')
plt.title('Feature Importance Comparison (Top 10)')
plt.legend()
plt.grid(True, alpha=0.3)

# AdaBoost confusion matrix
plt.subplot(2, 2, 3)
ada_cm = confusion_matrix(y_test, ada_predictions)
sns.heatmap(ada_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.title('AdaBoost Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Gradient Boosting confusion matrix
plt.subplot(2, 2, 4)
gb_cm = confusion_matrix(y_test, gb_predictions)
sns.heatmap(gb_cm, annot=True, fmt='d', cmap='Reds', 
            xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.title('Gradient Boosting Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('wine_boosting_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Error analysis
print("🔍 Error Analysis:")
print("-" * 40)

ada_errors = ada_predictions != y_test
gb_errors = gb_predictions != y_test

print(f"AdaBoost errors: {ada_errors.sum()}/{len(y_test)}")
print(f"Gradient Boosting errors: {gb_errors.sum()}/{len(y_test)}")

# Find samples where methods disagree
disagreement = ada_predictions != gb_predictions
print(f"Disagreement between methods: {disagreement.sum()}/{len(y_test)} samples")

if disagreement.sum() > 0:
    print("\nSamples where methods disagree:")
    disagreement_indices = np.where(disagreement)[0]
    for idx in disagreement_indices[:5]:  # Show first 5
        print(f"  Sample {idx}: True={wine.target_names[y_test[idx]]}, "
              f"AdaBoost={wine.target_names[ada_predictions[idx]]}, "
              f"GradientBoost={wine.target_names[gb_predictions[idx]]}")

print()
print("🎯 Key Takeaways:")
print("1. AdaBoost focuses on reweighting misclassified samples")
print("2. Gradient Boosting fits residual errors sequentially") 
print("3. Both methods combine weak learners into strong ensembles")
print("4. Feature importance patterns reveal different learning strategies")
print("5. Training progression shows how accuracy improves with more trees")
print(f"\n📁 Saved plot: wine_boosting_comparison.png")

print("\n🧠 Conceptual Differences:")
print("• AdaBoost: 'Let's focus more on the hard examples!'")
print("• Gradient Boosting: 'Let's predict what we got wrong!'")
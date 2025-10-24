#!/usr/bin/env python3
"""
Random Forest vs Extra-Trees Speed Comparison
==============================================

Demonstrates that Extra-Trees train faster than Random Forests
on a large dataset with many features.

Key Learning Points:
- Extra-Trees use random thresholds (no optimization)
- Random Forests optimize thresholds (more computation)
- Speed difference becomes significant with large datasets

Author: Course Materials - EAS 510 BAI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# Create output directory
os.makedirs('images/ensemble', exist_ok=True)

def create_large_separable_dataset(n_samples=50000, n_features=100, n_informative=80):
    """Create a large dataset with many features that's still separable"""
    print(f"🎯 Creating dataset: {n_samples:,} samples, {n_features} features...")
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,  # Most features are useful
        n_redundant=10,               # Some redundant features
        n_clusters_per_class=2,       # Make it separable but not trivial
        class_sep=1.5,               # Good separation between classes
        random_state=42
    )
    
    print(f"✅ Dataset created: {X.shape[0]:,} samples × {X.shape[1]} features")
    print(f"   Class distribution: {np.bincount(y)}")
    return X, y

def run_timing_experiment(X_train, X_test, y_train, y_test, n_estimators=100):
    """Run RandomForest vs ExtraTrees with detailed timing"""
    
    results = {}
    
    print(f"\n🌳 Training Random Forest ({n_estimators} trees)...")
    start_time = time.time()
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,           # Limit depth for fair comparison
        min_samples_split=5,
        n_jobs=-1,              # Use all cores
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    rf_train_time = time.time() - start_time
    
    # Get predictions and accuracy
    start_time = time.time()
    rf_pred = rf.predict(X_test)
    rf_predict_time = time.time() - start_time
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    results['RandomForest'] = {
        'train_time': rf_train_time,
        'predict_time': rf_predict_time,
        'accuracy': rf_accuracy,
        'model': rf
    }
    
    print(f"   Training time: {rf_train_time:.2f} seconds")
    print(f"   Prediction time: {rf_predict_time:.4f} seconds")
    print(f"   Accuracy: {rf_accuracy:.4f}")
    
    print(f"\n🚀 Training Extra-Trees ({n_estimators} trees)...")
    start_time = time.time()
    
    et = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=10,           # Same depth as RF
        min_samples_split=5,
        n_jobs=-1,              # Use all cores
        random_state=42
    )
    et.fit(X_train, y_train)
    
    et_train_time = time.time() - start_time
    
    # Get predictions and accuracy
    start_time = time.time()
    et_pred = et.predict(X_test)
    et_predict_time = time.time() - start_time
    et_accuracy = accuracy_score(y_test, et_pred)
    
    results['ExtraTrees'] = {
        'train_time': et_train_time,
        'predict_time': et_predict_time,
        'accuracy': et_accuracy,
        'model': et
    }
    
    print(f"   Training time: {et_train_time:.2f} seconds")
    print(f"   Prediction time: {et_predict_time:.4f} seconds")
    print(f"   Accuracy: {et_accuracy:.4f}")
    
    return results

def analyze_results(results):
    """Analyze and visualize the timing results"""
    
    rf_results = results['RandomForest']
    et_results = results['ExtraTrees']
    
    print(f"\n📊 SPEED COMPARISON RESULTS:")
    print(f"{'='*50}")
    
    # Training time comparison
    train_speedup = rf_results['train_time'] / et_results['train_time']
    print(f"Training Time:")
    print(f"   Random Forest: {rf_results['train_time']:.2f} seconds")
    print(f"   Extra-Trees:   {et_results['train_time']:.2f} seconds")
    print(f"   🚀 Extra-Trees is {train_speedup:.1f}x FASTER")
    
    # Prediction time comparison
    pred_speedup = rf_results['predict_time'] / et_results['predict_time']
    print(f"\nPrediction Time:")
    print(f"   Random Forest: {rf_results['predict_time']:.4f} seconds")
    print(f"   Extra-Trees:   {et_results['predict_time']:.4f} seconds")
    print(f"   🚀 Extra-Trees is {pred_speedup:.1f}x faster")
    
    # Accuracy comparison
    accuracy_diff = et_results['accuracy'] - rf_results['accuracy']
    print(f"\nAccuracy:")
    print(f"   Random Forest: {rf_results['accuracy']:.4f}")
    print(f"   Extra-Trees:   {et_results['accuracy']:.4f}")
    print(f"   📈 Difference: {accuracy_diff:+.4f}")
    
    return train_speedup, pred_speedup, accuracy_diff

def create_visualizations(results):
    """Create timing comparison visualizations"""
    
    rf_results = results['RandomForest']
    et_results = results['ExtraTrees']
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training time comparison
    methods = ['Random Forest', 'Extra-Trees']
    train_times = [rf_results['train_time'], et_results['train_time']]
    colors = ['#ff7f0e', '#2ca02c']
    
    bars1 = ax1.bar(methods, train_times, color=colors)
    ax1.set_ylabel('Training Time (seconds)')
    ax1.set_title('🌳 Training Time Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, time_val in zip(bars1, train_times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # Prediction time comparison
    pred_times = [rf_results['predict_time'], et_results['predict_time']]
    bars2 = ax2.bar(methods, pred_times, color=colors)
    ax2.set_ylabel('Prediction Time (seconds)')
    ax2.set_title('⚡ Prediction Time Comparison')
    ax2.grid(True, alpha=0.3)
    
    for bar, time_val in zip(bars2, pred_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                f'{time_val:.4f}s', ha='center', va='bottom', fontweight='bold')
    
    # Accuracy comparison
    accuracies = [rf_results['accuracy'], et_results['accuracy']]
    bars3 = ax3.bar(methods, accuracies, color=colors)
    ax3.set_ylabel('Accuracy')
    ax3.set_title('🎯 Accuracy Comparison')
    ax3.set_ylim(0.85, 1.0)  # Focus on the interesting range
    ax3.grid(True, alpha=0.3)
    
    for bar, acc_val in zip(bars3, accuracies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{acc_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Speed-up visualization
    train_speedup = rf_results['train_time'] / et_results['train_time']
    pred_speedup = rf_results['predict_time'] / et_results['predict_time']
    
    speedups = [train_speedup, pred_speedup]
    speedup_labels = ['Training\nSpeedup', 'Prediction\nSpeedup']
    bars4 = ax4.bar(speedup_labels, speedups, color=['#d62728', '#9467bd'])
    ax4.set_ylabel('Speedup Factor (X times faster)')
    ax4.set_title('🚀 Extra-Trees Speedup')
    ax4.grid(True, alpha=0.3)
    
    for bar, speedup_val in zip(bars4, speedups):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{speedup_val:.1f}x', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('images/ensemble/rf_vs_extratrees_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"💾 Visualization saved to: images/ensemble/rf_vs_extratrees_comparison.png")

def run_scalability_test(X, y, tree_counts=[50, 100, 200, 500]):
    """Test how the speed difference scales with number of trees"""
    
    print(f"\n📈 SCALABILITY TEST: How does speed difference change with more trees?")
    print(f"{'='*70}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scalability_results = {
        'n_trees': [],
        'rf_times': [],
        'et_times': [],
        'speedups': []
    }
    
    for n_trees in tree_counts:
        print(f"\n🌳 Testing with {n_trees} trees...")
        
        # Random Forest
        start_time = time.time()
        rf = RandomForestClassifier(n_estimators=n_trees, max_depth=8, n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train)
        rf_time = time.time() - start_time
        
        # Extra-Trees
        start_time = time.time()
        et = ExtraTreesClassifier(n_estimators=n_trees, max_depth=8, n_jobs=-1, random_state=42)
        et.fit(X_train, y_train)
        et_time = time.time() - start_time
        
        speedup = rf_time / et_time
        
        scalability_results['n_trees'].append(n_trees)
        scalability_results['rf_times'].append(rf_time)
        scalability_results['et_times'].append(et_time)
        scalability_results['speedups'].append(speedup)
        
        print(f"   RF: {rf_time:.1f}s | ET: {et_time:.1f}s | Speedup: {speedup:.1f}x")
    
    # Plot scalability results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(scalability_results['n_trees'], scalability_results['rf_times'], 
             'o-', label='Random Forest', linewidth=2, markersize=8)
    plt.plot(scalability_results['n_trees'], scalability_results['et_times'], 
             's--', label='Extra-Trees', linewidth=2, markersize=8)
    plt.xlabel('Number of Trees')
    plt.ylabel('Training Time (seconds)')
    plt.title('🌳 Training Time vs Number of Trees')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(scalability_results['n_trees'], scalability_results['speedups'], 
             '^-', color='red', linewidth=2, markersize=8)
    plt.xlabel('Number of Trees')
    plt.ylabel('Speedup Factor')
    plt.title('🚀 Extra-Trees Speedup vs Number of Trees')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/ensemble/scalability_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"💾 Scalability plot saved to: images/ensemble/scalability_comparison.png")
    
    return scalability_results

def main():
    """Main experiment function"""
    print("🚀 Random Forest vs Extra-Trees Speed Experiment")
    print("=" * 60)
    
    # Create large dataset
    X, y = create_large_separable_dataset(
        n_samples=50000,    # Large dataset
        n_features=100,     # Many features
        n_informative=80    # Most are useful
    )
    
    # Split the data
    print(f"\n📊 Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training set: {X_train.shape[0]:,} samples")
    print(f"   Test set: {X_test.shape[0]:,} samples")
    
    # Run main timing experiment
    results = run_timing_experiment(X_train, X_test, y_train, y_test, n_estimators=100)
    
    # Analyze results
    train_speedup, pred_speedup, accuracy_diff = analyze_results(results)
    
    # Create visualizations
    create_visualizations(results)
    
    # Run scalability test
    scalability_results = run_scalability_test(X, y)
    
    # Final summary
    print(f"\n🎯 EXPERIMENT SUMMARY:")
    print(f"{'='*50}")
    print(f"💡 Why Extra-Trees are faster:")
    print(f"   • Random Forest: Optimizes split thresholds")
    print(f"   • Extra-Trees: Uses random thresholds")
    print(f"   • Result: {train_speedup:.1f}x faster training!")
    print(f"\n🎯 Trade-offs:")
    print(f"   • Speed: Extra-Trees win by {train_speedup:.1f}x")
    print(f"   • Accuracy: Similar performance ({accuracy_diff:+.3f} difference)")
    print(f"   • Recommendation: Use Extra-Trees when speed matters!")

if __name__ == "__main__":
    main()
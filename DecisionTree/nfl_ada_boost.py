#!/usr/bin/env python3
"""
NFL AdaBoost Analysis
=====================

An in-class exercise for understanding AdaBoost ensemble methods using real NFL data.
Students will predict game outcomes using AdaBoost with decision stumps and compare
performance against single decision trees.

Learning Objectives:
- Understand how AdaBoost sequentially improves weak learners
- Compare single tree vs AdaBoost performance
- Analyze which features AdaBoost stumps choose
- Visualize sequential specialization on hard cases

Author: Course Materials - EAS 510 BAI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
try:
    # https://github.com/nflverse/nfl_data_py
    import nfl_data_py as nfl
    NFL_DATA_AVAILABLE = True
except ImportError:
    print("📝 Note: nfl_data_py not installed. Will use synthetic data for demonstration.")
    NFL_DATA_AVAILABLE = False
    nfl = None

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
Path('images/ensemble').mkdir(parents=True, exist_ok=True)

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class NFLAdaBoostAnalysis:
    """Class to handle NFL AdaBoost ensemble analysis"""
    
    def __init__(self, season=2024):
        """Initialize with specified season"""
        self.season = season
        self.team_data = None
        self.game_data = None
        self.features = None
        self.target = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.adaboost_model = None
        self.single_tree_model = None
        
    def load_data(self):
        """Load and process NFL data"""
        print(f"📊 Loading NFL {self.season} season data...")
        
        if not NFL_DATA_AVAILABLE:
            raise ImportError("nfl_data_py is required for live data. Install with: pip install nfl_data_py")
        
        try:
            # Load weekly player data
            print("📥 Loading weekly player data...")
            weekly_data = nfl.import_weekly_data([self.season])
            
            # Load team descriptions
            print("📥 Loading team descriptions...")
            team_data = nfl.import_team_desc()
            
            # Load game schedules
            print("📥 Loading game schedules...")
            schedules = nfl.import_schedules([self.season])
            
            print(f"✅ Loaded {len(weekly_data)} weekly records from {self.season} season")
            
            # Process the data for our AdaBoost analysis
            self.process_live_data(weekly_data, team_data, schedules)
            
        except Exception as e:
            print(f"❌ Error loading live NFL data: {e}")
            print("💡 Make sure you have nfl_data_py installed: pip install nfl_data_py")
            raise
    
    def process_live_data(self, weekly_data, team_data, schedules):
        """Process live NFL data for AdaBoost analysis"""
        print("🔧 Processing live NFL data...")
        
        # Filter for relevant positions and stats
        relevant_positions = ['QB', 'RB', 'WR', 'TE']
        weekly_filtered = weekly_data[weekly_data['position'].isin(relevant_positions)].copy()
        
        # Aggregate team-level statistics by week
        team_weekly_stats = []
        
        for week in weekly_filtered['week'].unique():
            week_data = weekly_filtered[weekly_filtered['week'] == week]
            
            for team in week_data['recent_team'].unique():
                team_week_data = week_data[week_data['recent_team'] == team]
                
                # Calculate team stats for this week
                team_stats = {
                    'week': week,
                    'team': team,
                    'passing_yards': team_week_data[team_week_data['position'] == 'QB']['passing_yards'].sum(),
                    'rushing_yards': team_week_data[team_week_data['position'].isin(['RB', 'QB'])]['rushing_yards'].sum(),
                    'receiving_yards': team_week_data[team_week_data['position'].isin(['WR', 'TE', 'RB'])]['receiving_yards'].sum(),
                    'fantasy_points': team_week_data['fantasy_points'].sum(),
                    'targets': team_week_data['targets'].sum(),
                    'carries': team_week_data['carries'].sum()
                }
                
                team_weekly_stats.append(team_stats)
        
        team_stats_df = pd.DataFrame(team_weekly_stats)
        
        # Merge with schedule data to get game results
        games_data = []
        
        for _, game in schedules.iterrows():
            if pd.isna(game.get('result', None)):
                continue  # Skip games not yet played
                
            # Home team perspective
            home_stats = team_stats_df[
                (team_stats_df['team'] == game['home_team']) & 
                (team_stats_df['week'] == game['week'])
            ]
            
            # Away team perspective  
            away_stats = team_stats_df[
                (team_stats_df['team'] == game['away_team']) & 
                (team_stats_df['week'] == game['week'])
            ]
            
            if len(home_stats) > 0 and len(away_stats) > 0:
                home_row = home_stats.iloc[0]
                away_row = away_stats.iloc[0]
                
                # Home team game record
                home_won = 1 if game['home_score'] > game['away_score'] else 0
                games_data.append({
                    'team': game['home_team'],
                    'opponent': game['away_team'],
                    'location': 'Home',
                    'result': home_won,
                    'score': game['home_score'],
                    'opp_score': game['away_score'],
                    'passing_yards': home_row['passing_yards'],
                    'rushing_yards': home_row['rushing_yards'],
                    'receiving_yards': home_row['receiving_yards'],
                    'total_yards': home_row['passing_yards'] + home_row['rushing_yards'] + home_row['receiving_yards'],
                    'fantasy_points': home_row['fantasy_points'],
                    'week': game['week']
                })
                
                # Away team game record
                away_won = 1 if game['away_score'] > game['home_score'] else 0
                games_data.append({
                    'team': game['away_team'],
                    'opponent': game['home_team'],
                    'location': 'Away',
                    'result': away_won,
                    'score': game['away_score'],
                    'opp_score': game['home_score'],
                    'passing_yards': away_row['passing_yards'],
                    'rushing_yards': away_row['rushing_yards'],
                    'receiving_yards': away_row['receiving_yards'],
                    'total_yards': away_row['passing_yards'] + away_row['rushing_yards'] + away_row['receiving_yards'],
                    'fantasy_points': away_row['fantasy_points'],
                    'week': game['week']
                })
        
        self.game_data = pd.DataFrame(games_data)
        print(f"✅ Processed {len(self.game_data)} team game records from live data")
        
        # Store additional data
        self.team_data = team_data
        self.schedules = schedules
    
    def prepare_features(self):
        """Prepare features for AdaBoost ensemble"""
        print("🔧 Engineering features for AdaBoost analysis...")
        
        # Create feature matrix
        features_df = self.game_data.copy()
        
        # Binary feature: home field advantage
        features_df['is_home'] = (features_df['location'] == 'Home').astype(int)
        
        # Offensive efficiency features
        features_df['passing_efficiency'] = features_df['passing_yards'] / features_df['passing_yards'].mean()
        features_df['rushing_efficiency'] = features_df['rushing_yards'] / features_df['rushing_yards'].mean()
        features_df['receiving_efficiency'] = features_df['receiving_yards'] / features_df['receiving_yards'].mean()
        features_df['total_offense'] = features_df['total_yards']
        
        # Overall offensive performance
        features_df['fantasy_performance'] = features_df['fantasy_points']
        
        # Select final features
        feature_columns = [
            'is_home', 'total_offense', 'fantasy_performance', 
            'passing_yards', 'rushing_yards', 'receiving_yards', 
            'passing_efficiency', 'rushing_efficiency'
        ]
        
        self.features = features_df[feature_columns]
        self.target = features_df['result']
        
        print(f"📈 Created {len(feature_columns)} features:")
        for i, col in enumerate(feature_columns, 1):
            print(f"  {i}. {col}")
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, test_size=test_size, 
            random_state=random_state, stratify=self.target
        )
        
        print(f"📚 Training set: {len(self.X_train)} games")
        print(f"🧪 Test set: {len(self.X_test)} games")
        print(f"⚖️ Win rate - Train: {self.y_train.mean():.1%}, Test: {self.y_test.mean():.1%}")
    
    def train_models(self, n_estimators=50, learning_rate=1.0):
        """Train both AdaBoost and single decision tree models"""
        print(f"🚀 Training AdaBoost ensemble ({n_estimators} stumps)...")
        
        # AdaBoost with decision stumps
        self.adaboost_model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),  # Decision stumps!
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
        
        # Single decision tree for comparison
        self.single_tree_model = DecisionTreeClassifier(
            max_depth=None,  # Let it grow deep
            random_state=42
        )
        
        # Train both models
        self.adaboost_model.fit(self.X_train, self.y_train)
        self.single_tree_model.fit(self.X_train, self.y_train)
        
        # Evaluate performance
        ada_train_acc = self.adaboost_model.score(self.X_train, self.y_train)
        ada_test_acc = self.adaboost_model.score(self.X_test, self.y_test)
        tree_train_acc = self.single_tree_model.score(self.X_train, self.y_train)
        tree_test_acc = self.single_tree_model.score(self.X_test, self.y_test)
        
        print(f"🎯 AdaBoost - Train: {ada_train_acc:.3f}, Test: {ada_test_acc:.3f}")
        print(f"🌳 Single Tree - Train: {tree_train_acc:.3f}, Test: {tree_test_acc:.3f}")
        
        if ada_test_acc > tree_test_acc:
            improvement = (ada_test_acc - tree_test_acc) * 100
            print(f"🎉 AdaBoost wins by {improvement:.1f} percentage points!")
        else:
            print("🤔 Single tree performed better this time.")
    
    def analyze_stump_choices(self, n_stumps_to_show=10):
        """Analyze which features the first several stumps chose"""
        if self.adaboost_model is None:
            print("❌ No trained AdaBoost model found!")
            return
        
        print(f"\n🔍 AdaBoost Stump Analysis (First {n_stumps_to_show} stumps):")
        print("=" * 70)
        
        feature_names = self.features.columns
        stump_info = []
        
        for i in range(min(n_stumps_to_show, len(self.adaboost_model.estimators_))):
            stump = self.adaboost_model.estimators_[i]
            weight = self.adaboost_model.estimator_weights_[i]
            
            # Get the split information
            feature_idx = stump.tree_.feature[0]  # Root node feature
            threshold = stump.tree_.threshold[0]  # Root node threshold
            feature_name = feature_names[feature_idx]
            
            stump_info.append({
                'stump': i + 1,
                'feature': feature_name,
                'threshold': threshold,
                'weight': weight
            })
            
            print(f"Stump {i+1:2d}: {feature_name:20s} > {threshold:8.2f} (α = {weight:.3f})")
        
        # Count feature usage
        feature_usage = {}
        for info in stump_info:
            feature = info['feature']
            feature_usage[feature] = feature_usage.get(feature, 0) + 1
        
        print(f"\n📊 Feature Usage in First {n_stumps_to_show} Stumps:")
        for feature, count in sorted(feature_usage.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature:20s}: {count} times")
        
        return stump_info
    
    def compare_performance(self):
        """Comprehensive performance comparison"""
        if self.adaboost_model is None or self.single_tree_model is None:
            print("❌ Models not trained yet!")
            return
        
        # Get predictions
        ada_pred = self.adaboost_model.predict(self.X_test)
        tree_pred = self.single_tree_model.predict(self.X_test)
        
        print("\n=== 🆚 ADABOOST vs SINGLE TREE COMPARISON ===")
        
        # Accuracy comparison
        ada_acc = accuracy_score(self.y_test, ada_pred)
        tree_acc = accuracy_score(self.y_test, tree_pred)
        
        print(f"AdaBoost Accuracy: {ada_acc:.4f}")
        print(f"Single Tree Accuracy: {tree_acc:.4f}")
        print(f"Improvement: {(ada_acc - tree_acc)*100:+.1f} percentage points")
        
        # Cross-validation comparison
        ada_cv = cross_val_score(self.adaboost_model, self.X_train, self.y_train, cv=5)
        tree_cv = cross_val_score(self.single_tree_model, self.X_train, self.y_train, cv=5)
        
        print(f"\nCross-Validation Scores:")
        print(f"AdaBoost: {ada_cv.mean():.3f} ± {ada_cv.std():.3f}")
        print(f"Single Tree: {tree_cv.mean():.3f} ± {tree_cv.std():.3f}")
        
        # Detailed reports
        print(f"\n=== AdaBoost Classification Report ===")
        print(classification_report(self.y_test, ada_pred, target_names=['Loss', 'Win']))
        
        print(f"\n=== Single Tree Classification Report ===")
        print(classification_report(self.y_test, tree_pred, target_names=['Loss', 'Win']))
    
    def plot_feature_importance_comparison(self):
        """Compare feature importance between AdaBoost and single tree"""
        if self.adaboost_model is None or self.single_tree_model is None:
            print("❌ Models not trained yet!")
            return
        
        # Get feature importances
        ada_importance = self.adaboost_model.feature_importances_
        tree_importance = self.single_tree_model.feature_importances_
        feature_names = self.features.columns
        
        # Create comparison dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'AdaBoost': ada_importance,
            'Single_Tree': tree_importance
        })
        
        # Plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # AdaBoost importance
        ada_sorted = importance_df.sort_values('AdaBoost', ascending=True)
        bars1 = ax1.barh(ada_sorted['Feature'], ada_sorted['AdaBoost'], color='skyblue')
        ax1.set_title('🚀 AdaBoost Feature Importance')
        ax1.set_xlabel('Importance Score')
        ax1.grid(axis='x', alpha=0.3)
        
        # Single tree importance
        tree_sorted = importance_df.sort_values('Single_Tree', ascending=True)
        bars2 = ax2.barh(tree_sorted['Feature'], tree_sorted['Single_Tree'], color='lightcoral')
        ax2.set_title('🌳 Single Tree Feature Importance')
        ax2.set_xlabel('Importance Score')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/ensemble/adaboost_feature_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print numerical comparison
        print("\n📊 Feature Importance Comparison:")
        print("-" * 50)
        importance_df_sorted = importance_df.sort_values('AdaBoost', ascending=False)
        for _, row in importance_df_sorted.iterrows():
            print(f"{row['Feature']:20s}: AdaBoost={row['AdaBoost']:.3f}, Tree={row['Single_Tree']:.3f}")
    
    def plot_learning_curve(self, max_estimators=100):
        """Plot how AdaBoost performance improves with more estimators"""
        print("📈 Analyzing learning curve...")
        
        estimator_range = range(1, max_estimators + 1, 5)
        train_scores = []
        test_scores = []
        
        for n_est in estimator_range:
            ada_temp = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1),
                n_estimators=n_est,
                learning_rate=1.0,
                random_state=42
            )
            ada_temp.fit(self.X_train, self.y_train)
            
            train_score = ada_temp.score(self.X_train, self.y_train)
            test_score = ada_temp.score(self.X_test, self.y_test)
            
            train_scores.append(train_score)
            test_scores.append(test_score)
        
        # Plot learning curve
        plt.figure(figsize=(10, 6))
        plt.plot(estimator_range, train_scores, label='Training Accuracy', marker='o')
        plt.plot(estimator_range, test_scores, label='Test Accuracy', marker='s')
        
        # Add single tree baseline
        tree_test_acc = self.single_tree_model.score(self.X_test, self.y_test)
        plt.axhline(y=tree_test_acc, color='red', linestyle='--', 
                   label=f'Single Tree Baseline ({tree_test_acc:.3f})')
        
        plt.xlabel('Number of Estimators (Stumps)')
        plt.ylabel('Accuracy')
        plt.title('🚀 AdaBoost Learning Curve: Sequential Improvement')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('images/ensemble/adaboost_learning_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_game(self, is_home=1, total_offense=400, fantasy_performance=50, 
                     passing_yards=280, rushing_yards=120, receiving_yards=200, 
                     passing_efficiency=1.0, rushing_efficiency=1.0):
        """Predict outcome using both models"""
        if self.adaboost_model is None or self.single_tree_model is None:
            print("❌ Models not trained yet!")
            return
        
        # Create input array
        game_input = np.array([[is_home, total_offense, fantasy_performance, 
                              passing_yards, rushing_yards, receiving_yards, 
                              passing_efficiency, rushing_efficiency]])
        
        # Get predictions
        ada_pred = self.adaboost_model.predict(game_input)[0]
        ada_prob = self.adaboost_model.predict_proba(game_input)[0]
        tree_pred = self.single_tree_model.predict(game_input)[0]
        tree_prob = self.single_tree_model.predict_proba(game_input)[0]
        
        print("🔮 Game Prediction Comparison:")
        print(f"   📍 Home field: {'Yes' if is_home else 'No'}")
        print(f"   📊 Total offense: {total_offense} yards")
        print(f"   🏆 Fantasy performance: {fantasy_performance} points")
        print()
        print(f"🚀 AdaBoost Prediction:")
        print(f"   🎯 Outcome: {'WIN' if ada_pred else 'LOSS'}")
        print(f"   📈 Win probability: {ada_prob[1]:.1%}")
        print()
        print(f"🌳 Single Tree Prediction:")
        print(f"   🎯 Outcome: {'WIN' if tree_pred else 'LOSS'}")
        print(f"   📈 Win probability: {tree_prob[1]:.1%}")
        print()
        print(f"🤝 Agreement: {'Yes' if ada_pred == tree_pred else 'No'}")


def main():
    """Main function to run the NFL AdaBoost analysis"""
    print("🚀 NFL AdaBoost vs Single Tree Analysis")
    print("=" * 60)
    
    # Initialize analysis
    nfl_analysis = NFLAdaBoostAnalysis(season=2024)
    
    # Load and prepare data
    nfl_analysis.load_data()
    nfl_analysis.prepare_features()
    nfl_analysis.split_data()
    
    # Train both models
    nfl_analysis.train_models(n_estimators=50, learning_rate=1.0)
    
    # Analyze what stumps chose
    nfl_analysis.analyze_stump_choices(n_stumps_to_show=10)
    
    # Compare performance
    nfl_analysis.compare_performance()
    
    # Visualizations
    print("\n📊 Generating comparison visualizations...")
    nfl_analysis.plot_feature_importance_comparison()
    nfl_analysis.plot_learning_curve()
    
    # Example prediction
    print("\n🔮 Example Predictions:")
    nfl_analysis.predict_game(
        is_home=1,
        total_offense=420,
        fantasy_performance=55,
        passing_yards=300,
        rushing_yards=120,
        receiving_yards=200,
        passing_efficiency=1.2,
        rushing_efficiency=1.1
    )
    
    print("\n✅ AdaBoost analysis complete! Ready for class discussion.")
    print("\n🚀 Key Takeaways:")
    print("   • AdaBoost combines many weak stumps into a strong classifier")
    print("   • Each stump specializes on errors from previous ensemble")
    print("   • Sequential learning focuses on hardest-to-classify games")
    print("   • Later stumps often get higher weights than early ones")
    print("   • Feature selection happens naturally through stump choices")


if __name__ == "__main__":
    main()
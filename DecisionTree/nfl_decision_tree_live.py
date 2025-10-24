#!/usr/bin/env python3
"""
NFL Bagging Analysis
====================

An in-class exercise for understanding bagging ensemble methods using real NFL data.
Students will predict game outcomes using Bootstrap Aggregating with decision trees.

Learning Objectives:
- Understand how bagging reduces overfitting
- Compare single tree vs ensemble performance
- Interpret feature importance across multiple trees
- Visualize ensemble decision boundaries

Author: Course Materials - EAS 510 BAI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import BaggingClassifier
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

class NFLBaggingAnalysis:
    """Class to handle NFL bagging ensemble analysis"""
    
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
        self.bagging_model = None
        
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
            
            # Peek at the data structures
            print(f"📊 Weekly data (PLAYER-ORIENTED) shape: {weekly_data.shape}")
            print(f"🔍 Weekly columns: {list(weekly_data.columns)}")
            print(f"📈 Weekly sample:")
            print(weekly_data[['player_name', 'position', 'recent_team', 'week', 'fantasy_points']].head(3))
            
            print(f"\n🏈 Team data shape: {team_data.shape}")
            print(f"🔍 Team columns: {list(team_data.columns)}")
            print(f"📈 Team sample:")
            print(team_data.head(3))
            
            print(f"\n📅 Schedule data (GAME-ORIENTED) shape: {schedules.shape}")
            print(f"🔍 Schedule columns: {list(schedules.columns)}")
            print(f"📈 Schedule sample:")
            print(schedules[['week', 'home_team', 'away_team', 'home_score', 'away_score']].head(3))
            
            # Process the data for our bagging analysis
            self.process_live_data(weekly_data, team_data, schedules)
            
        except Exception as e:
            print(f"❌ Error loading live NFL data: {e}")
            print("💡 Make sure you have nfl_data_py installed: pip install nfl_data_py")
            raise
    
    def process_live_data(self, weekly_data, team_data, schedules):
        """Process live NFL data for bagging analysis"""
        print("🔧 Processing live NFL data...")
        
        # Filter for relevant positions and stats
        # Focus on key skill position players that affect game outcomes
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
        # Convert schedule data to have both team perspectives
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
        """Prepare features for bagging ensemble"""
        print("🔧 Engineering features for bagging analysis...")
        
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
        
        # Score differential (points scored vs allowed)
        features_df['score_differential'] = features_df['score'] - features_df['opp_score']
        
        # Select final features
        feature_columns = [
            'is_home', 'total_offense', 'fantasy_performance', 
            'passing_yards', 'rushing_yards', 'receiving_yards', 'passing_efficiency', 'rushing_efficiency'
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
    
    def train_bagging_ensemble(self, n_estimators=100, max_samples=100, max_depth=None, min_samples_leaf=5):
        """Train bagging ensemble model"""
        print(f"🎒 Training bagging ensemble ({n_estimators} trees)...")
        
        # Create base decision tree
        base_tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        # Create bagging classifier
        self.bagging_model = BaggingClassifier(
            estimator=base_tree,
            n_estimators=n_estimators,
            max_samples=max_samples,
            n_jobs=-1,
            random_state=42
        )
        
        self.bagging_model.fit(self.X_train, self.y_train)
        
        # Evaluate performance
        train_accuracy = self.bagging_model.score(self.X_train, self.y_train)
        test_accuracy = self.bagging_model.score(self.X_test, self.y_test)
        
        print(f"🎯 Training Accuracy: {train_accuracy:.3f}")
        print(f"🎯 Test Accuracy: {test_accuracy:.3f}")
        
        return self.bagging_model
    
    def analyze_ensemble_performance(self):
        """Comprehensive performance analysis of the bagging ensemble"""
        if self.bagging_model is None:
            print("❌ No trained model found. Train a bagging ensemble first!")
            return
        
        y_pred = self.bagging_model.predict(self.X_test)
        y_pred_proba = self.bagging_model.predict_proba(self.X_test)
        
        print("=== 🎒 BAGGING ENSEMBLE RESULTS ===")
        print(f"Number of estimators: {self.bagging_model.n_estimators}")
        print(f"Max samples per estimator: {self.bagging_model.max_samples}")
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        
        print("\n=== CONFUSION MATRIX ===")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        print("\n=== DETAILED CLASSIFICATION REPORT ===")
        print(classification_report(self.y_test, y_pred, target_names=['Loss', 'Win']))
        
        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Predicted Loss', 'Predicted Win'],
                   yticklabels=['Actual Loss', 'Actual Win'])
        plt.title('🎒 Bagging Ensemble Confusion Matrix')
        plt.tight_layout()
        plt.savefig('images/ensemble/bagging_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_feature_importance(self):
        """Analyze feature importance across the ensemble"""
        if self.bagging_model is None:
            print("❌ No trained model found. Train a bagging ensemble first!")
            return
        
        # Calculate average feature importance across all trees
        importances = np.mean([tree.feature_importances_ for tree in self.bagging_model.estimators_], axis=0)
        feature_names = self.features.columns
        
        print("\n=== ENSEMBLE FEATURE IMPORTANCES ===")
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(importance_df.iterrows(), 1):
            print(f"  {i}. {row['feature']}: {row['importance']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        importance_df_sorted = importance_df.sort_values('importance', ascending=True)
        bars = plt.barh(importance_df_sorted['feature'], importance_df_sorted['importance'])
        
        # Color bars by importance
        colors = plt.cm.viridis(importance_df_sorted['importance'] / importance_df_sorted['importance'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.title('🔍 Ensemble Feature Importance in NFL Game Prediction', fontsize=14, pad=20)
        plt.xlabel('Average Importance Score Across All Trees')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('images/ensemble/bagging_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_decision_boundary_2d(self):
        """Plot decision boundary using the two most important features"""
        if self.bagging_model is None:
            print("❌ No trained model found. Train a bagging ensemble first!")
            return
        
        # Get the two most important features
        importances = np.mean([tree.feature_importances_ for tree in self.bagging_model.estimators_], axis=0)
        top_features_idx = np.argsort(importances)[-2:]
        feature_names = self.features.columns[top_features_idx]
        
        # Extract 2D data
        X_2d = self.X_train.iloc[:, top_features_idx].values
        y_2d = self.y_train.values
        
        # Train a new bagging model on just these 2 features for visualization
        bagging_2d = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
            n_estimators=100,
            max_samples=100,
            n_jobs=-1,
            random_state=42
        )
        bagging_2d.fit(X_2d, y_2d)
        
        # Create mesh for decision boundary
        h = 0.02
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        plt.figure(figsize=(15, 5))
        
        # Plot decision boundary
        plt.subplot(1, 3, 1)
        Z = bagging_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdYlBu, edgecolors='black')
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title('🎒 Bagging Ensemble - Decision Boundary')
        plt.colorbar(scatter)
        
        # Plot prediction probabilities
        plt.subplot(1, 3, 2)
        Z_proba = bagging_2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z_proba = Z_proba.reshape(xx.shape)
        contour = plt.contourf(xx, yy, Z_proba, levels=20, alpha=0.8, cmap=plt.cm.RdYlBu)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdYlBu, edgecolors='black')
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title('🎒 Win Probability Heatmap')
        plt.colorbar(contour)
        
        # Plot individual tree boundaries (sample of 5 trees)
        plt.subplot(1, 3, 3)
        for i, tree in enumerate(bagging_2d.estimators_[:5]):
            Z_tree = tree.predict(np.c_[xx.ravel(), yy.ravel()])
            Z_tree = Z_tree.reshape(xx.shape)
            plt.contour(xx, yy, Z_tree, alpha=0.3, colors=['red', 'blue'][i % 2])
        
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdYlBu, edgecolors='black')
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title('Individual Trees (Sample of 5)')
        
        plt.tight_layout()
        plt.savefig('images/ensemble/bagging_decision_boundary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Decision boundary plot saved to: images/ensemble/bagging_decision_boundary.png")
        print(f"Using features: {feature_names[0]} and {feature_names[1]}")
    
    def compare_ensemble_sizes(self, n_estimators_list=[1, 5, 10, 25, 50, 100, 200]):
        """Compare performance across different ensemble sizes"""
        print("📊 Comparing ensemble sizes to understand bagging benefits...")
        
        results = []
        for n_est in n_estimators_list:
            # Create bagging classifier with varying ensemble size
            bagging = BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=None, random_state=42),
                n_estimators=n_est,
                max_samples=100,
                n_jobs=-1,
                random_state=42
            )
            
            # Use cross-validation for robust estimates
            cv_scores = cross_val_score(bagging, self.X_train, self.y_train, cv=5)
            
            # Also get test score
            bagging.fit(self.X_train, self.y_train)
            test_score = bagging.score(self.X_test, self.y_test)
            
            results.append({
                'n_estimators': n_est,
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_score': test_score
            })
            
            print(f"  {n_est:>3} trees: CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}, Test={test_score:.3f}")
        
        # Plot results
        results_df = pd.DataFrame(results)
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(results_df['n_estimators'], results_df['cv_score'], 
                    yerr=results_df['cv_std'], label='Cross-Validation', 
                    marker='o', capsize=5)
        plt.plot(results_df['n_estimators'], results_df['test_score'], 
                label='Test Set', marker='s', linestyle='--')
        
        plt.xlabel('Number of Trees in Ensemble')
        plt.ylabel('Accuracy')
        plt.title('🎒 Bagging Performance vs Ensemble Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('images/ensemble/ensemble_size_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_game(self, is_home=1, total_offense=400, fantasy_performance=50, 
                     passing_yards=280, rushing_yards=120, receiving_yards=200, 
                     passing_efficiency=1.0, rushing_efficiency=1.0):
        """Predict outcome for a specific game scenario using the ensemble"""
        if self.bagging_model is None:
            print("❌ No trained model found. Train a bagging ensemble first!")
            return
        
        # Create input array
        game_input = np.array([[is_home, total_offense, fantasy_performance, 
                              passing_yards, rushing_yards, receiving_yards, 
                              passing_efficiency, rushing_efficiency]])
        
        # Get prediction and probability
        prediction = self.bagging_model.predict(game_input)[0]
        probability = self.bagging_model.predict_proba(game_input)[0]
        
        print("🔮 Ensemble Game Prediction:")
        print(f"   📍 Home field: {'Yes' if is_home else 'No'}")
        print(f"   📊 Total offense: {total_offense} yards")
        print(f"   🏆 Fantasy performance: {fantasy_performance} points")
        print(f"   ✈️  Passing: {passing_yards} yards")
        print(f"   🏃 Rushing: {rushing_yards} yards")
        print(f"   🎯 Receiving: {receiving_yards} yards")
        print()
        print(f"   🎯 Predicted outcome: {'WIN' if prediction else 'LOSS'}")
        print(f"   📈 Win probability: {probability[1]:.1%}")
        print(f"   📉 Loss probability: {probability[0]:.1%}")
        print(f"   🎒 Based on {self.bagging_model.n_estimators} decision trees")


def main():
    """Main function to run the NFL bagging analysis"""
    print("🎒 NFL Bagging Ensemble Analysis")
    print("=" * 50)
    
    # Initialize analysis
    nfl_analysis = NFLBaggingAnalysis(season=2024)
    
    # Load and prepare data
    nfl_analysis.load_data()
    nfl_analysis.prepare_features()
    nfl_analysis.split_data()
    
    # Train bagging ensemble
    nfl_analysis.train_bagging_ensemble(n_estimators=100, max_samples=100)
    
    # Comprehensive analysis
    print("\n📊 Generating ensemble analysis...")
    nfl_analysis.analyze_ensemble_performance()
    nfl_analysis.analyze_feature_importance()
    
    # Decision boundary visualization
    print("\n🎨 Creating decision boundary visualization...")
    nfl_analysis.plot_decision_boundary_2d()
    
    # Example prediction
    print("\n🔮 Example Ensemble Prediction:")
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
    
    # Compare different ensemble sizes
    print("\n📈 Ensemble Size Analysis:")
    nfl_analysis.compare_ensemble_sizes()
    
    print("\n✅ Bagging analysis complete! Ready for class discussion.")
    print("\n🎒 Key Takeaways:")
    print("   • Bagging reduces overfitting by averaging multiple trees")
    print("   • Each tree sees a different bootstrap sample of the data")
    print("   • Ensemble predictions are more stable than single trees")
    print("   • Feature importance is averaged across all trees")


if __name__ == "__main__":
    main()
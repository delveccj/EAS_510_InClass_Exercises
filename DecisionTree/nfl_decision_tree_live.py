#!/usr/bin/env python3
"""
NFL Decision Tree Analysis
==========================

An in-class exercise for understanding decision trees using real NFL data.
Students will predict game outcomes based on team statistics.

Learning Objectives:
- Understand how decision trees split on features
- Interpret tree decisions in a real-world context
- Visualize and analyze feature importance
- Explore overfitting and tree depth effects

Author: Course Materials - EAS 510 BAI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
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
Path('images/decision_tree').mkdir(parents=True, exist_ok=True)

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class NFLDecisionTreeAnalysis:
    """Class to handle NFL decision tree analysis"""
    
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
        self.tree_model = None
        
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
            
            # Process the data for our decision tree
            self.process_live_data(weekly_data, team_data, schedules)
            
        except Exception as e:
            print(f"❌ Error loading live NFL data: {e}")
            print("💡 Make sure you have nfl_data_py installed: pip install nfl_data_py")
            raise
    
    def process_live_data(self, weekly_data, team_data, schedules):
        """Process live NFL data for decision tree analysis"""
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
        """Prepare features for decision tree"""
        print("🔧 Engineering features for decision tree...")
        
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
            'passing_yards', 'rushing_yards', 'receiving_yards', 'passing_efficiency'
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
    
    def train_tree(self, max_depth=3, min_samples_leaf=5):
        """Train decision tree model"""
        print(f"🌳 Training decision tree (max_depth={max_depth})...")
        
        self.tree_model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        self.tree_model.fit(self.X_train, self.y_train)
        
        # Evaluate performance
        train_accuracy = self.tree_model.score(self.X_train, self.y_train)
        test_accuracy = self.tree_model.score(self.X_test, self.y_test)
        
        print(f"🎯 Training Accuracy: {train_accuracy:.3f}")
        print(f"🎯 Test Accuracy: {test_accuracy:.3f}")
        
        return self.tree_model
    
    def visualize_tree(self, figsize=(15, 10)):
        """Visualize the decision tree"""
        if self.tree_model is None:
            print("❌ No trained model found. Train a tree first!")
            return
        
        plt.figure(figsize=figsize)
        plot_tree(
            self.tree_model,
            feature_names=self.features.columns,
            class_names=['Loss', 'Win'],
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.title("🏈 NFL Game Outcome Decision Tree", fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()
        plt.savefig('images/decision_tree/decision_tree.png', dpi=300, bbox_inches='tight')

    
    def analyze_feature_importance(self):
        """Analyze and visualize feature importance"""
        if self.tree_model is None:
            print("❌ No trained model found. Train a tree first!")
            return
        
        importance = self.tree_model.feature_importances_
        feature_names = self.features.columns
        
        # Create DataFrame for easy plotting
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        # Plot
        plt.figure(figsize=(10, 6))
        bars = plt.barh(importance_df['feature'], importance_df['importance'])
        
        # Color bars by importance
        colors = plt.cm.viridis(importance_df['importance'] / importance_df['importance'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.title('🔍 Feature Importance in NFL Game Prediction', fontsize=14, pad=20)
        plt.xlabel('Importance Score')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()
        plt.savefig('images/decision_tree/importance.png', dpi=300, bbox_inches='tight')
        
        print("📊 Feature Importance Ranking:")
        for i, (_, row) in enumerate(importance_df.sort_values('importance', ascending=False).iterrows(), 1):
            print(f"  {i}. {row['feature']}: {row['importance']:.3f}")
    
    def print_tree_rules(self):
        """Print human-readable tree rules"""
        if self.tree_model is None:
            print("❌ No trained model found. Train a tree first!")
            return
        
        tree_rules = export_text(
            self.tree_model,
            feature_names=list(self.features.columns)
        )
        
        print("📋 Decision Tree Rules:")
        print("=" * 50)
        print(tree_rules)
    
    def predict_game(self, is_home=1, total_offense=400, fantasy_performance=50, 
                     passing_yards=280, rushing_yards=120, receiving_yards=200, passing_efficiency=0.043):
        """Predict outcome for a specific game scenario"""
        if self.tree_model is None:
            print("❌ No trained model found. Train a tree first!")
            return
        
        # Create input array
        game_input = np.array([[is_home, total_offense, fantasy_performance, 
                              passing_yards, rushing_yards, receiving_yards, passing_efficiency]])
        
        # Get prediction and probability
        prediction = self.tree_model.predict(game_input)[0]
        probability = self.tree_model.predict_proba(game_input)[0]
        
        print("🔮 Game Prediction:")
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
    
    def compare_tree_depths(self, depths=[1, 2, 3, 4, 5, None]):
        """Compare performance across different tree depths"""
        print("📊 Comparing tree depths to understand overfitting...")
        
        results = []
        for depth in depths:
            tree = DecisionTreeClassifier(
                max_depth=depth, 
                min_samples_leaf=5, 
                random_state=42
            )
            
            # Use cross-validation for more robust estimates
            cv_scores = cross_val_score(tree, self.X_train, self.y_train, cv=5)
            
            # Also get test score
            tree.fit(self.X_train, self.y_train)
            test_score = tree.score(self.X_test, self.y_test)
            
            depth_str = str(depth) if depth else "Unlimited"
            results.append({
                'depth': depth_str,
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_score': test_score
            })
            
            print(f"  Depth {depth_str:>9}: CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}, Test={test_score:.3f}")
        
        # Plot results
        results_df = pd.DataFrame(results)
        
        plt.figure(figsize=(10, 6))
        x_pos = range(len(results_df))
        
        plt.errorbar(x_pos, results_df['cv_score'], yerr=results_df['cv_std'], 
                    label='Cross-Validation', marker='o', capsize=5)
        plt.plot(x_pos, results_df['test_score'], 
                label='Test Set', marker='s', linestyle='--')
        
        plt.xlabel('Tree Depth')
        plt.ylabel('Accuracy')
        plt.title('🌳 Tree Performance vs Depth: Finding the Sweet Spot')
        plt.xticks(x_pos, results_df['depth'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        plt.savefig('images/decision_tree/tree.png', dpi=300, bbox_inches='tight')

    
    def confusion_matrix_analysis(self):
        """Analyze model performance with confusion matrix"""
        if self.tree_model is None:
            print("❌ No trained model found. Train a tree first!")
            return
        
        y_pred = self.tree_model.predict(self.X_test)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Predicted Loss', 'Predicted Win'],
                   yticklabels=['Actual Loss', 'Actual Win'])
        plt.title('🏈 Confusion Matrix: Model Performance')
        plt.tight_layout()
        plt.show()
        plt.savefig('images/decision_tree/confusion_matrix.png', dpi=300, bbox_inches='tight')
        
        # Detailed report
        print("📊 Detailed Performance Report:")
        print(classification_report(self.y_test, y_pred, 
                                  target_names=['Loss', 'Win']))


def main():
    """Main function to run the NFL decision tree analysis"""
    print("🏈 NFL Decision Tree Analysis")
    print("=" * 50)
    
    # Initialize analysis
    nfl_analysis = NFLDecisionTreeAnalysis(season=2024)
    
    # Load and prepare data
    nfl_analysis.load_data()
    nfl_analysis.prepare_features()
    nfl_analysis.split_data()
    
    # Train initial model
    nfl_analysis.train_tree(max_depth=3)
    
    # Visualizations and analysis
    print("\n🎨 Generating visualizations...")
    nfl_analysis.visualize_tree()
    nfl_analysis.analyze_feature_importance()
    
    # Print interpretable rules
    print("\n📋 Tree Rules:")
    nfl_analysis.print_tree_rules()
    
    # Example prediction
    print("\n🔮 Example Prediction:")
    nfl_analysis.predict_game(
        is_home=1,
        total_offense=420,
        fantasy_performance=55,
        passing_yards=300,
        rushing_yards=120,
        receiving_yards=200,
        passing_efficiency=0.08
    )
    
    # Compare different tree depths
    print("\n📊 Overfitting Analysis:")
    nfl_analysis.compare_tree_depths()
    
    # Confusion matrix
    print("\n🎯 Performance Analysis:")
    nfl_analysis.confusion_matrix_analysis()
    
    print("\n✅ Analysis complete! Ready for class discussion.")


if __name__ == "__main__":
    main()
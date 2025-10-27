#!/usr/bin/env python3
"""
NFL AdaBoost vs Gradient Boosting Analysis
==========================================

An in-class exercise comparing AdaBoost and Gradient Boosting ensemble methods 
using real NFL data. Students will predict game outcomes and compare how these
different boosting strategies approach sequential learning.

Learning Objectives:
- Compare AdaBoost vs Gradient Boosting on real sports data
- Understand different approaches to sequential error correction
- Analyze feature importance patterns between methods
- Visualize performance progression and convergence
- Learn when to use each boosting method

Author: Course Materials - EAS 510 BAI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
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

class NFLBoostingComparison:
    """Class to handle NFL AdaBoost vs Gradient Boosting analysis"""
    
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
        self.gradient_boosting_model = None
        
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
            
            # Process the data for our boosting analysis
            self.process_live_data(weekly_data, team_data, schedules)
            
        except Exception as e:
            print(f"❌ Error loading live NFL data: {e}")
            print("💡 Make sure you have nfl_data_py installed: pip install nfl_data_py")
            raise
    
    def process_live_data(self, weekly_data, team_data, schedules):
        """Process live NFL data for boosting analysis"""
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
        """Prepare features for boosting ensemble analysis"""
        print("🔧 Engineering features for boosting comparison...")
        
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
    
    def train_models(self, n_estimators=50):
        """Train both AdaBoost and Gradient Boosting models"""
        print(f"🚀 Training ensemble methods ({n_estimators} estimators each)...")
        
        # AdaBoost with decision stumps
        print("   🔸 AdaBoost with decision stumps...")
        self.adaboost_model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),  # Decision stumps!
            n_estimators=n_estimators,
            learning_rate=1.0,  # Higher learning rate for AdaBoost
            random_state=42
        )
        
        # Gradient Boosting with shallow trees
        print("   🔹 Gradient Boosting with shallow trees...")
        self.gradient_boosting_model = GradientBoostingClassifier(
            max_depth=3,  # Shallow trees for GB
            n_estimators=n_estimators,
            learning_rate=0.1,  # Lower learning rate for GB
            random_state=42
        )
        
        # Train both models
        self.adaboost_model.fit(self.X_train, self.y_train)
        self.gradient_boosting_model.fit(self.X_train, self.y_train)
        
        # Evaluate performance
        ada_train_acc = self.adaboost_model.score(self.X_train, self.y_train)
        ada_test_acc = self.adaboost_model.score(self.X_test, self.y_test)
        gb_train_acc = self.gradient_boosting_model.score(self.X_train, self.y_train)
        gb_test_acc = self.gradient_boosting_model.score(self.X_test, self.y_test)
        
        print(f"🎯 AdaBoost - Train: {ada_train_acc:.3f}, Test: {ada_test_acc:.3f}")
        print(f"🌟 Gradient Boosting - Train: {gb_train_acc:.3f}, Test: {gb_test_acc:.3f}")
        
        if gb_test_acc > ada_test_acc:
            improvement = (gb_test_acc - ada_test_acc) * 100
            print(f"🎉 Gradient Boosting wins by {improvement:.1f} percentage points!")
        elif ada_test_acc > gb_test_acc:
            improvement = (ada_test_acc - gb_test_acc) * 100
            print(f"🚀 AdaBoost wins by {improvement:.1f} percentage points!")
        else:
            print("🤝 It's a tie!")
    
    def analyze_algorithm_differences(self):
        """Analyze key differences between the algorithms"""
        if self.adaboost_model is None or self.gradient_boosting_model is None:
            print("❌ Models not trained yet!")
            return
        
        print(f"\n🔍 Algorithm Difference Analysis:")
        print("=" * 70)
        
        # AdaBoost analysis
        print("\n🚀 AdaBoost Approach:")
        print("   • Uses decision stumps (max_depth=1)")
        print("   • Reweights samples based on errors")
        print("   • Focuses on misclassified examples")
        print(f"   • Learning rate: {self.adaboost_model.learning_rate}")
        
        feature_names = self.features.columns
        print("\n   First 5 stumps chosen:")
        for i in range(min(5, len(self.adaboost_model.estimators_))):
            stump = self.adaboost_model.estimators_[i]
            weight = self.adaboost_model.estimator_weights_[i]
            feature_idx = stump.tree_.feature[0]
            threshold = stump.tree_.threshold[0]
            feature_name = feature_names[feature_idx]
            print(f"     Stump {i+1}: {feature_name} > {threshold:.2f} (α = {weight:.3f})")
        
        # Gradient Boosting analysis
        print(f"\n🌟 Gradient Boosting Approach:")
        print("   • Uses shallow trees (max_depth=3)")
        print("   • Fits residual errors from previous predictions")
        print("   • Sequential error correction")
        print(f"   • Learning rate: {self.gradient_boosting_model.learning_rate}")
        print(f"   • Each tree predicts residuals from ensemble so far")
        
        print(f"\n📊 Key Differences Summary:")
        print("   AdaBoost: Changes the PROBLEM (sample weights)")
        print("   Gradient Boosting: Changes the TARGET (residual errors)")
    
    def compare_performance(self):
        """Comprehensive performance comparison"""
        if self.adaboost_model is None or self.gradient_boosting_model is None:
            print("❌ Models not trained yet!")
            return
        
        # Get predictions
        ada_pred = self.adaboost_model.predict(self.X_test)
        gb_pred = self.gradient_boosting_model.predict(self.X_test)
        
        print("\n=== 🆚 ADABOOST vs GRADIENT BOOSTING COMPARISON ===")
        
        # Accuracy comparison
        ada_acc = accuracy_score(self.y_test, ada_pred)
        gb_acc = accuracy_score(self.y_test, gb_pred)
        
        print(f"AdaBoost Accuracy: {ada_acc:.4f}")
        print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")
        print(f"Difference: {(gb_acc - ada_acc)*100:+.1f} percentage points")
        
        # Cross-validation comparison
        ada_cv = cross_val_score(self.adaboost_model, self.X_train, self.y_train, cv=5)
        gb_cv = cross_val_score(self.gradient_boosting_model, self.X_train, self.y_train, cv=5)
        
        print(f"\nCross-Validation Scores:")
        print(f"AdaBoost: {ada_cv.mean():.3f} ± {ada_cv.std():.3f}")
        print(f"Gradient Boosting: {gb_cv.mean():.3f} ± {gb_cv.std():.3f}")
        
        # Detailed reports
        print(f"\n=== AdaBoost Classification Report ===")
        print(classification_report(self.y_test, ada_pred, target_names=['Loss', 'Win']))
        
        print(f"\n=== Gradient Boosting Classification Report ===")
        print(classification_report(self.y_test, gb_pred, target_names=['Loss', 'Win']))
        
        # Agreement analysis
        agreement = (ada_pred == gb_pred).sum()
        disagreement = len(ada_pred) - agreement
        print(f"\n🤝 Model Agreement:")
        print(f"   Agree on: {agreement}/{len(ada_pred)} predictions ({agreement/len(ada_pred):.1%})")
        print(f"   Disagree on: {disagreement}/{len(ada_pred)} predictions ({disagreement/len(ada_pred):.1%})")
    
    def plot_comprehensive_comparison(self):
        """Create comprehensive comparison visualizations"""
        if self.adaboost_model is None or self.gradient_boosting_model is None:
            print("❌ Models not trained yet!")
            return
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Learning progression
        plt.subplot(2, 3, 1)
        self._plot_learning_progression()
        
        # 2. Feature importance comparison
        plt.subplot(2, 3, 2)
        self._plot_feature_importance()
        
        # 3. Confusion matrices
        plt.subplot(2, 3, 3)
        self._plot_confusion_matrices()
        
        # 4. Prediction agreement
        plt.subplot(2, 3, 4)
        self._plot_prediction_agreement()
        
        # 5. Training vs Test performance
        plt.subplot(2, 3, 5)
        self._plot_overfitting_analysis()
        
        # 6. Prediction confidence
        plt.subplot(2, 3, 6)
        self._plot_prediction_confidence()
        
        plt.tight_layout()
        plt.savefig('images/ensemble/nfl_boosting_comprehensive_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_learning_progression(self):
        """Plot how each algorithm improves over time"""
        # Get staged predictions for both algorithms
        ada_staged = list(self.adaboost_model.staged_score(self.X_test, self.y_test))
        gb_staged = list(self.gradient_boosting_model.staged_score(self.X_test, self.y_test))
        
        estimators = range(1, len(ada_staged) + 1)
        
        plt.plot(estimators, ada_staged, 'b-', label='AdaBoost', linewidth=2)
        plt.plot(estimators, gb_staged, 'r-', label='Gradient Boosting', linewidth=2)
        plt.xlabel('Number of Estimators')
        plt.ylabel('Test Accuracy')
        plt.title('🚀 Learning Progression')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    def _plot_feature_importance(self):
        """Compare feature importance between methods"""
        ada_importance = self.adaboost_model.feature_importances_
        gb_importance = self.gradient_boosting_model.feature_importances_
        
        feature_names = self.features.columns
        x = np.arange(len(feature_names))
        width = 0.35
        
        plt.barh(x - width/2, ada_importance, width, label='AdaBoost', alpha=0.8)
        plt.barh(x + width/2, gb_importance, width, label='Gradient Boosting', alpha=0.8)
        
        plt.yticks(x, feature_names, fontsize=8)
        plt.xlabel('Feature Importance')
        plt.title('📊 Feature Importance')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    def _plot_confusion_matrices(self):
        """Plot confusion matrices side by side"""
        ada_pred = self.adaboost_model.predict(self.X_test)
        gb_pred = self.gradient_boosting_model.predict(self.X_test)
        
        ada_cm = confusion_matrix(self.y_test, ada_pred)
        gb_cm = confusion_matrix(self.y_test, gb_pred)
        
        # Combined confusion matrix visualization
        combined_cm = np.array([[ada_cm[0,0], ada_cm[0,1]], [gb_cm[0,0], gb_cm[0,1]]])
        
        sns.heatmap(ada_cm, annot=True, fmt='d', cmap='Blues', alpha=0.7,
                   xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
        plt.title('🎯 AdaBoost Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
    
    def _plot_prediction_agreement(self):
        """Plot where models agree vs disagree"""
        ada_pred = self.adaboost_model.predict(self.X_test)
        gb_pred = self.gradient_boosting_model.predict(self.X_test)
        
        agreement = ada_pred == gb_pred
        agreement_rate = agreement.mean()
        
        labels = ['Disagree', 'Agree']
        sizes = [1-agreement_rate, agreement_rate]
        colors = ['lightcoral', 'lightgreen']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('🤝 Model Agreement')
    
    def _plot_overfitting_analysis(self):
        """Analyze overfitting for both models"""
        ada_train_score = self.adaboost_model.score(self.X_train, self.y_train)
        ada_test_score = self.adaboost_model.score(self.X_test, self.y_test)
        gb_train_score = self.gradient_boosting_model.score(self.X_train, self.y_train)
        gb_test_score = self.gradient_boosting_model.score(self.X_test, self.y_test)
        
        models = ['AdaBoost', 'Gradient Boosting']
        train_scores = [ada_train_score, gb_train_score]
        test_scores = [ada_test_score, gb_test_score]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, train_scores, width, label='Training', alpha=0.8)
        plt.bar(x + width/2, test_scores, width, label='Test', alpha=0.8)
        
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('📈 Training vs Test Performance')
        plt.xticks(x, models)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    def _plot_prediction_confidence(self):
        """Plot prediction confidence distributions"""
        ada_proba = self.adaboost_model.predict_proba(self.X_test)
        gb_proba = self.gradient_boosting_model.predict_proba(self.X_test)
        
        # Max probability for each prediction (confidence)
        ada_confidence = np.max(ada_proba, axis=1)
        gb_confidence = np.max(gb_proba, axis=1)
        
        plt.hist(ada_confidence, alpha=0.7, label='AdaBoost', bins=20)
        plt.hist(gb_confidence, alpha=0.7, label='Gradient Boosting', bins=20)
        
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('🎯 Prediction Confidence')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    def predict_game_comparison(self, is_home=1, total_offense=400, fantasy_performance=50, 
                              passing_yards=280, rushing_yards=120, receiving_yards=200, 
                              passing_efficiency=1.0, rushing_efficiency=1.0):
        """Compare predictions from both models"""
        if self.adaboost_model is None or self.gradient_boosting_model is None:
            print("❌ Models not trained yet!")
            return
        
        # Create input array
        game_input = np.array([[is_home, total_offense, fantasy_performance, 
                              passing_yards, rushing_yards, receiving_yards, 
                              passing_efficiency, rushing_efficiency]])
        
        # Get predictions from both models
        ada_pred = self.adaboost_model.predict(game_input)[0]
        ada_prob = self.adaboost_model.predict_proba(game_input)[0]
        gb_pred = self.gradient_boosting_model.predict(game_input)[0]
        gb_prob = self.gradient_boosting_model.predict_proba(game_input)[0]
        
        print("🔮 Game Prediction Comparison:")
        print(f"   📍 Home field: {'Yes' if is_home else 'No'}")
        print(f"   📊 Total offense: {total_offense} yards")
        print(f"   🏆 Fantasy performance: {fantasy_performance} points")
        print()
        print(f"🚀 AdaBoost Prediction:")
        print(f"   🎯 Outcome: {'WIN' if ada_pred else 'LOSS'}")
        print(f"   📈 Win probability: {ada_prob[1]:.1%}")
        print()
        print(f"🌟 Gradient Boosting Prediction:")
        print(f"   🎯 Outcome: {'WIN' if gb_pred else 'LOSS'}")
        print(f"   📈 Win probability: {gb_prob[1]:.1%}")
        print()
        print(f"🤝 Agreement: {'Yes' if ada_pred == gb_pred else 'No'}")
        
        # Explain the difference in approach
        print(f"\n🧠 Why might they differ?")
        print(f"   AdaBoost: Focused on reweighting hard-to-classify games")
        print(f"   Gradient Boosting: Fitted residual errors sequentially")


def main():
    """Main function to run the NFL boosting comparison analysis"""
    print("🚀 NFL AdaBoost vs Gradient Boosting Analysis")
    print("=" * 70)
    
    # Initialize analysis
    nfl_analysis = NFLBoostingComparison(season=2024)
    
    # Load and prepare data
    nfl_analysis.load_data()
    nfl_analysis.prepare_features()
    nfl_analysis.split_data()
    
    # Train both models
    nfl_analysis.train_models(n_estimators=50)
    
    # Analyze algorithmic differences
    nfl_analysis.analyze_algorithm_differences()
    
    # Compare performance
    nfl_analysis.compare_performance()
    
    # Create visualizations
    print("\n📊 Generating comprehensive comparison visualizations...")
    nfl_analysis.plot_comprehensive_comparison()
    
    # Example predictions
    print("\n🔮 Example Prediction Comparison:")
    nfl_analysis.predict_game_comparison(
        is_home=1,
        total_offense=420,
        fantasy_performance=55,
        passing_yards=300,
        rushing_yards=120,
        receiving_yards=200,
        passing_efficiency=1.2,
        rushing_efficiency=1.1
    )
    
    print("\n✅ Boosting comparison analysis complete! Ready for class discussion.")
    print("\n🎯 Key Takeaways:")
    print("   • AdaBoost focuses on sample reweighting (changes the problem)")
    print("   • Gradient Boosting fits residual errors (changes the target)")
    print("   • Both combine weak learners into strong ensembles")
    print("   • Different learning rates optimize different objectives")
    print("   • Feature importance patterns reveal algorithmic strategies")
    print("   • Performance depends on data characteristics and noise levels")


if __name__ == "__main__":
    main()
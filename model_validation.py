import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

from models import MLModelTrainer

class ModelValidation:
    """
    Model validation with confusion matrix and feature importance analysis
    """
    
    def __init__(self, trainer):
        """
        Initialize with trained MLModelTrainer
        
        Args:
            trainer: MLModelTrainer instance with trained models
        """
        self.trainer = trainer
        self.feature_names = self.trainer.X.columns.tolist() if hasattr(self.trainer.X, 'columns') else None
        
    def validate_all_models(self):
        """
        Validate all trained models with confusion matrix and feature importance
        """
        print("=" * 80)
        print("MODEL VALIDATION REPORT")
        print("=" * 80)
        
        results = {}
        
        for model_name, model in self.trainer.models.items():
            print(f"\n{'='*20} {model_name.upper()} {'='*20}")
            
            # Get predictions
            y_pred = model.predict(self.trainer.X_test)
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(self.trainer.X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(self.trainer.y_test, y_pred, y_pred_proba)
            
            # Display metrics
            self._display_metrics(model_name, metrics)
            
            # Plot confusion matrix
            self._plot_confusion_matrix(self.trainer.y_test, y_pred, model_name)
            
            # Analyze feature importance
            feature_importance = self._analyze_feature_importance(model, model_name)
            
            # Store results
            results[model_name] = {
                'metrics': metrics,
                'predictions': y_pred,
                'feature_importance': feature_importance
            }
            
        return results
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add AUC for binary classification with probability predictions
        if y_pred_proba is not None and y_pred_proba.shape[1] == 2:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except:
                metrics['auc'] = None
        
        return metrics
    
    def _display_metrics(self, model_name, metrics):
        """Display performance metrics"""
        print(f"\nPerformance Metrics for {model_name}:")
        print("-" * 40)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        if metrics.get('auc') is not None:
            print(f"AUC:       {metrics['auc']:.4f}")
    
    def _plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot confusion matrix for the model"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(y_true), 
                   yticklabels=np.unique(y_true))
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add accuracy information
        accuracy = np.trace(cm) / np.sum(cm)
        plt.text(0.5, -0.1, f'Accuracy: {accuracy:.4f}', 
                ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.show()
        
        # Print confusion matrix details
        print(f"\nConfusion Matrix for {model_name}:")
        print(cm)
        
        # Calculate per-class metrics
        if len(np.unique(y_true)) == 2:  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            print(f"True Negatives:  {tn}")
            print(f"False Positives: {fp}")
            print(f"False Negatives: {fn}")
            print(f"True Positives:  {tp}")
            
            if tp + fn > 0:
                sensitivity = tp / (tp + fn)
                print(f"Sensitivity (Recall): {sensitivity:.4f}")
            
            if tn + fp > 0:
                specificity = tn / (tn + fp)
                print(f"Specificity: {specificity:.4f}")
    
    def _analyze_feature_importance(self, model, model_name):
        """Analyze and plot feature importance"""
        feature_importance = None
        
        # Method 1: Built-in feature importance (tree-based models)
        if hasattr(model, 'feature_importances_'):
            feature_importance = self._get_builtin_importance(model, model_name)
        
        # Method 2: Coefficient importance (linear models)
        elif hasattr(model, 'coef_'):
            feature_importance = self._get_coefficient_importance(model, model_name)
        
        # Method 3: Permutation importance (works for all models)
        else:
            feature_importance = self._get_permutation_importance(model, model_name)
        
        return feature_importance
    
    def _get_builtin_importance(self, model, model_name):
        """Get built-in feature importance for tree-based models"""
        importances = model.feature_importances_
        
        # Create feature importance dataframe
        if self.feature_names:
            feature_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
        else:
            feature_df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(importances))],
                'importance': importances
            }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Feature Importances for {model_name} (Built-in):")
        print(feature_df.head(10).to_string(index=False))
        
        # Plot feature importance
        self._plot_feature_importance(feature_df, model_name, "Built-in Feature Importance")
        
        return feature_df
    
    def _get_coefficient_importance(self, model, model_name):
        """Get coefficient importance for linear models"""
        if hasattr(model, 'coef_'):
            coefs = model.coef_
            if len(coefs.shape) > 1:
                coefs = coefs[0]  # For binary classification
            
            # Create feature importance dataframe
            if self.feature_names:
                feature_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'coefficient': coefs,
                    'abs_coefficient': np.abs(coefs)
                }).sort_values('abs_coefficient', ascending=False)
            else:
                feature_df = pd.DataFrame({
                    'feature': [f'feature_{i}' for i in range(len(coefs))],
                    'coefficient': coefs,
                    'abs_coefficient': np.abs(coefs)
                }).sort_values('abs_coefficient', ascending=False)
            
            print(f"\nTop 10 Feature Coefficients for {model_name}:")
            print(feature_df.head(10)[['feature', 'coefficient']].to_string(index=False))
            
            # Plot coefficients
            self._plot_coefficients(feature_df, model_name)
            
            return feature_df
    
    def _get_permutation_importance(self, model, model_name):
        """Get permutation importance (works for all models)"""
        try:
            perm_importance = permutation_importance(
                model, self.trainer.X_test, self.trainer.y_test,
                n_repeats=5, random_state=42, n_jobs=-1
            )
            
            # Create feature importance dataframe
            if self.feature_names:
                feature_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': perm_importance.importances_mean,
                    'std': perm_importance.importances_std
                }).sort_values('importance', ascending=False)
            else:
                feature_df = pd.DataFrame({
                    'feature': [f'feature_{i}' for i in range(len(perm_importance.importances_mean))],
                    'importance': perm_importance.importances_mean,
                    'std': perm_importance.importances_std
                }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Permutation Feature Importances for {model_name}:")
            print(feature_df.head(10).to_string(index=False))
            
            # Plot permutation importance
            self._plot_permutation_importance(feature_df, model_name)
            
            return feature_df
            
        except Exception as e:
            print(f"Could not calculate permutation importance for {model_name}: {e}")
            return None
    
    def _plot_feature_importance(self, feature_df, model_name, title):
        """Plot feature importance bar chart"""
        plt.figure(figsize=(12, 8))
        top_features = feature_df.head(15)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'{title} - {model_name}')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, v in enumerate(top_features['importance']):
            plt.text(v, i, f' {v:.3f}', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_coefficients(self, feature_df, model_name):
        """Plot coefficient values"""
        plt.figure(figsize=(12, 8))
        top_features = feature_df.head(15)
        
        # Color bars based on positive/negative coefficients
        colors = ['red' if x < 0 else 'blue' for x in top_features['coefficient']]
        
        plt.barh(range(len(top_features)), top_features['coefficient'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Coefficient Value')
        plt.title(f'Feature Coefficients - {model_name}')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, v in enumerate(top_features['coefficient']):
            plt.text(v, i, f' {v:.3f}', va='center')
        
        # Add legend
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.text(0.02, 0.98, 'Blue = Positive\nRed = Negative', 
                transform=plt.gca().transAxes, va='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def _plot_permutation_importance(self, feature_df, model_name):
        """Plot permutation importance with error bars"""
        plt.figure(figsize=(12, 8))
        top_features = feature_df.head(15)
        
        plt.barh(range(len(top_features)), top_features['importance'], 
                xerr=top_features['std'], capsize=3)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Permutation Importance')
        plt.title(f'Permutation Feature Importance - {model_name}')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, v in enumerate(top_features['importance']):
            plt.text(v, i, f' {v:.3f}', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def compare_models_summary(self, results):
        """Create a summary comparison of all models"""
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, result in results.items():
            metrics = result['metrics']
            row = {
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1']
            }
            if metrics.get('auc') is not None:
                row['AUC'] = metrics['auc']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        print("\nModel Performance Comparison:")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Identify best model
        best_model = comparison_df.iloc[0]['Model']
        best_accuracy = comparison_df.iloc[0]['Accuracy']
        
        print(f"\n🏆 Best Performing Model: {best_model}")
        print(f"   Accuracy: {best_accuracy:.4f}")
        
        return comparison_df

def main():
    """
    Main validation function
    """
    print("Starting Model Validation...")
    
    # Load and train models
    trainer = MLModelTrainer('data/ds1302.csv')
    trainer.explore_data()
    trainer.prepare_data('dead')
    trainer.train_classification_models(use_grid_search=True)
    
    # Create validator and run validation
    validator = ModelValidation(trainer)
    results = validator.validate_all_models()
    
    # Generate comparison summary
    comparison_df = validator.compare_models_summary(results)
    
    print("\nValidation completed successfully!")

if __name__ == "__main__":
    main()
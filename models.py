import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

# Clustering Models
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

import warnings
warnings.filterwarnings('ignore')

class MLModelTrainer:
    """
    A comprehensive ML model trainer for CSV data analysis
    """
    
    def __init__(self, data_path=None, df=None):
        """
        Initialize the model trainer
        
        Args:
            data_path (str): Path to CSV file
            df (DataFrame): Pre-loaded pandas DataFrame
        """
        if data_path:
            self.df = self.load_data(data_path)
        elif df is not None:
            self.df = df.copy()
        else:
            self.df = None
            
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
    
    def load_data(self, data_path):
        """Load data from CSV file"""
        self.df = pd.read_csv(data_path)
        # Replace specific values with NaN
        self.df.replace([99, 99.1, 99.2, 99.3, -9, -9.2, -9.1, -9.3], np.nan, inplace=True)

        # drop irrelavant columns
        self.df = self.df.drop(columns=["pseudoid","pseudoccn", "yeartx", "anc", "dwoancrel", "intxanc", "plt", "dwoplt", "intxplt", "agvhd24", "dwoagvhd24", "agvhd34", "dwoagvhd34", "intxagvhd34", "intxagvhd24", "cgvhd" , "dwocgvhd", "intxcgvhd", "dfs", "trm", "rel", "intxrel", "intxsurv", "pgf"])
        #drop columns with too many missing values
        self.df = self.df.drop(columns=["dnrage_urd", "dagegp", "rbcreduc13", "buffycp13", "rbcreduc4", "buffycp4", "plasmarmv"])

        # for row in self.df["racegp"].index:
        #     print(row)
        #     if row == 99:
        #         self.df["racegp"][row] = np.nan
        
        print(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def explore_data(self):
        """Basic data exploration"""
        if self.df is None:
            print("No data loaded")
            return
        
        print("=" * 50)
        print("DATA EXPLORATION")
        print("=" * 50)
        print(f"Shape: {self.df.shape}")
        print(f"\nData types:")
        print(self.df.dtypes)
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        print(f"\nBasic statistics:")
        print(self.df.describe())
    
    def prepare_data(self, target_column, test_size=0.2, scale_features=True):
        """
        Prepare data for training
        
        Args:
            target_column (str): Name of the target column
            test_size (float): Proportion of test data
            scale_features (bool): Whether to scale features
        """
        if self.df is None:
            raise ValueError("No data loaded")

        
        
        # drop everything but drabomatch and target column
       # self.df = self.df[[target_column, 'drabomatch']]
        
        # self.df.drop()
        self.df = self.df.dropna()
        
        print(f"Data after cleaning: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print(self.df)
        
        # Separate features and target
        self.X = self.df.drop(columns=[target_column])
        self.y = self.df[target_column]
        
        
        # Handle categorical variables
        # categorical_cols = self.X.select_dtypes(include=['object']).columns
        # if len(categorical_cols) > 0:
        #     print(f"Encoding categorical variables: {list(categorical_cols)}")
        #     self.X = pd.get_dummies(self.X, columns=categorical_cols, drop_first=True)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        
        # Scale features if requested
        if scale_features:
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
    
    def get_classification_models(self):
        """Get dictionary of classification models with parameters"""
        return {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42),
                'params': {
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'svm': {
                'model': SVC(random_state=42),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            },
            'knn': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                }
            },
            'naive_bayes': {
                'model': GaussianNB(),
                'params': {}
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'neural_network': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'alpha': [0.001, 0.01, 0.1],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
    
    def get_regression_models(self):
        """Get dictionary of regression models with parameters"""
        return {
            'linear_regression': {
                'model': LinearRegression(),
                'params': {}
            },
            'ridge': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': [0.1, 1, 10, 100]
                }
            },
            'lasso': {
                'model': Lasso(random_state=42),
                'params': {
                    'alpha': [0.1, 1, 10, 100]
                }
            },
            'elastic_net': {
                'model': ElasticNet(random_state=42),
                'params': {
                    'alpha': [0.1, 1, 10],
                    'l1_ratio': [0.1, 0.5, 0.9]
                }
            },
            'random_forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'svr': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            },
            'knn': {
                'model': KNeighborsRegressor(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                }
            },
            'decision_tree': {
                'model': DecisionTreeRegressor(random_state=42),
                'params': {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'neural_network': {
                'model': MLPRegressor(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'alpha': [0.001, 0.01, 0.1],
                    'learning_rate': ['constant', 'adaptive']
                }
            },
        }
    
    def train_classification_models(self, use_grid_search=False, cv_folds=5):
        """
        Train all classification models
        
        Args:
            use_grid_search (bool): Whether to use GridSearchCV for hyperparameter tuning
            cv_folds (int): Number of cross-validation folds
        """
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first")
        
        models = self.get_classification_models()
        
        print("Training Classification Models...")
        print("=" * 50)
        
        for name, model_info in models.items():
            print(f"Training {name}...")
            
            if use_grid_search and model_info['params']:
                # Use GridSearchCV for hyperparameter tuning
                grid = GridSearchCV(
                    model_info['model'], 
                    model_info['params'], 
                    cv=cv_folds,
                    scoring='f1',
                    n_jobs=-1
                )
                grid.fit(self.X_train, self.y_train)
                best_model = grid.best_estimator_
                print(f"Best params for {name}: {grid.best_params_}")
            else:
                # Use default parameters
                best_model = model_info['model']
                best_model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = best_model.predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            cv_scores = cross_val_score(best_model, self.X_train, self.y_train, cv=cv_folds)
            
            # Store results
            self.models[name] = best_model
            self.results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print("-" * 30)
    
    def train_regression_models(self, use_grid_search=False, cv_folds=5):
        """
        Train all regression models
        
        Args:
            use_grid_search (bool): Whether to use GridSearchCV for hyperparameter tuning
            cv_folds (int): Number of cross-validation folds
        """
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first")
        
        models = self.get_regression_models()
        
        print("Training Regression Models...")
        print("=" * 50)
        
        for name, model_info in models.items():
            print(f"Training {name}...")
            
            if use_grid_search and model_info['params']:
                # Use GridSearchCV for hyperparameter tuning
                grid = GridSearchCV(
                    model_info['model'], 
                    model_info['params'], 
                    cv=cv_folds,
                    scoring='r2',
                    n_jobs=-1
                )
                grid.fit(self.X_train, self.y_train)
                best_model = grid.best_estimator_
                print(f"Best params for {name}: {grid.best_params_}")
            else:
                # Use default parameters
                best_model = model_info['model']
                best_model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = best_model.predict(self.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Store results
            self.models[name] = best_model
            self.results[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"R²: {r2:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print("-" * 30)
    
    def compare_models(self, task_type='classification'):
        """
        Compare model performance
        
        Args:
            task_type (str): 'classification' or 'regression'
        """
        if not self.results:
            print("No models trained yet")
            return
        
        print(f"\nMODEL COMPARISON ({task_type.upper()})")
        print("=" * 60)
        
        if task_type == 'classification':
            # Sort by accuracy
            sorted_models = sorted(
                self.results.items(), 
                key=lambda x: x[1]['accuracy'], 
                reverse=True
            )
            
            print(f"{'Model':<20} {'Accuracy':<10} {'CV Mean':<10} {'CV Std':<10}")
            print("-" * 60)
            
            for name, metrics in sorted_models:
                print(f"{name:<20} {metrics['accuracy']:<10.4f} "
                      f"{metrics['cv_mean']:<10.4f} {metrics['cv_std']:<10.4f}")
                
        else:  # regression
            # Sort by R²
            sorted_models = sorted(
                self.results.items(), 
                key=lambda x: x[1]['r2'], 
                reverse=True
            )
            
            print(f"{'Model':<20} {'R²':<10} {'RMSE':<10} {'MAE':<10}")
            print("-" * 60)
            
            for name, metrics in sorted_models:
                print(f"{name:<20} {metrics['r2']:<10.4f} "
                      f"{metrics['rmse']:<10.4f} {metrics['mae']:<10.4f}")
    
    def get_best_model(self, task_type='classification'):
        """Get the best performing model"""
        if not self.results:
            return None
        
        if task_type == 'classification':
            best_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        else:
            best_name = max(self.results.keys(), key=lambda x: self.results[x]['r2'])
        
        return best_name, self.models[best_name]
    
    def predict_new_data(self, new_data, model_name=None):
        """
        Make predictions on new data
        
        Args:
            new_data (DataFrame or array): New data to predict
            model_name (str): Name of the model to use (uses best if None)
        """
        if model_name is None:
            # Use the best model
            model_name, model = self.get_best_model()
        else:
            model = self.models.get(model_name)
            if model is None:
                raise ValueError(f"Model '{model_name}' not found")
        
        # Preprocess new data (scale if needed)
        if hasattr(self.scaler, 'scale_'):
            new_data = self.scaler.transform(new_data)
        
        predictions = model.predict(new_data)
        return predictions

# Clustering Models
class ClusteringModels:
    """Unsupervised clustering models"""
    
    def __init__(self, data):
        self.data = data
        self.models = {}
        self.results = {}
    
    def kmeans_clustering(self, n_clusters_range=(2, 10)):
        """K-Means clustering with elbow method"""
        from sklearn.metrics import silhouette_score
        
        inertias = []
        silhouette_scores = []
        K_range = range(n_clusters_range[0], n_clusters_range[1] + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.data)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.data, labels))
        
        # Find best k using silhouette score
        best_k = K_range[np.argmax(silhouette_scores)]
        best_kmeans = KMeans(n_clusters=best_k, random_state=42)
        best_labels = best_kmeans.fit_predict(self.data)
        
        self.models['kmeans'] = best_kmeans
        self.results['kmeans'] = {
            'labels': best_labels,
            'n_clusters': best_k,
            'silhouette_score': max(silhouette_scores),
            'inertia': best_kmeans.inertia_
        }
        
        return best_labels, best_k
    
    def dbscan_clustering(self, eps=0.5, min_samples=5):
        """DBSCAN clustering"""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.data)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        self.models['dbscan'] = dbscan
        self.results['dbscan'] = {
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise
        }
        
        return labels

# Example usage functions
def example_classification():
    """Example usage for classification"""
    # Load data
    trainer = MLModelTrainer()
    trainer.load_data('your_data.csv')
    
    # Explore data
    trainer.explore_data()
    
    # Prepare data (assuming 'target' is your target column)
    trainer.prepare_data('target_column_name')
    
    # Train models
    trainer.train_classification_models(use_grid_search=True)
    
    # Compare models
    trainer.compare_models('classification')
    
    # Get best model
    best_name, best_model = trainer.get_best_model('classification')
    print(f"Best model: {best_name}")

def example_regression():
    """Example usage for regression"""
    # Load data
    trainer = MLModelTrainer()
    trainer.load_data('your_data.csv')
    
    # Prepare data
    trainer.prepare_data('target_column_name')
    
    # Train models
    trainer.train_regression_models(use_grid_search=True)
    
    # Compare models
    trainer.compare_models('regression')

if __name__ == "__main__":
    print("ML Models module loaded successfully!")
    print("Use MLModelTrainer class for supervised learning")
    print("Use ClusteringModels class for unsupervised learning")
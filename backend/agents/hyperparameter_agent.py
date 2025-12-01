"""
Agent 4: Hyperparameter Tuning & Prediction Agent
Performs hyperparameter tuning and makes predictions
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    make_scorer, accuracy_score, r2_score,
    mean_squared_error, mean_absolute_error,
    precision_score, recall_score, f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


class HyperparameterTuningAgent:
    """
    Agent for hyperparameter tuning and final predictions
    """

    def __init__(self):
        self.static_dir = Path("backend/static/plots")
        self.static_dir.mkdir(parents=True, exist_ok=True)
        self.tuned_model = None
        self.best_params = None
        self.cv_results = None

    def tune_and_predict(
        self,
        model,
        model_name: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        problem_type: str,
        tuning_method: str = "grid",
        cv_folds: int = 5,
        n_iter: int = 20,
        job_id: str = ""
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning and make predictions

        Args:
            model: Trained model to tune
            model_name: Name of the model
            X_train, X_test: Feature data
            y_train, y_test: Target data
            problem_type: 'classification' or 'regression'
            tuning_method: 'grid' or 'random'
            cv_folds: Number of CV folds
            n_iter: Number of iterations for random search
            job_id: Job identifier

        Returns:
            Dictionary with tuning results and predictions
        """
        # Get parameter grid for the model
        param_grid = self._get_param_grid(model_name, tuning_method)

        if not param_grid:
            return {
                "status": "skipped",
                "message": f"No hyperparameter grid defined for {model_name}",
                "using_default_model": True
            }

        # Setup scorer
        if problem_type == "classification":
            scorer = make_scorer(accuracy_score)
        else:
            scorer = make_scorer(r2_score)

        # Perform hyperparameter search
        if tuning_method == "grid":
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv_folds,
                scoring=scorer,
                n_jobs=-1,
                verbose=0
            )
        else:  # random
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv_folds,
                scoring=scorer,
                n_jobs=-1,
                verbose=0,
                random_state=42
            )

        # Fit the search
        try:
            search.fit(X_train, y_train)
            self.tuned_model = search.best_estimator_
            self.best_params = search.best_params_
            self.cv_results = search.cv_results_
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "message": f"Hyperparameter tuning failed for {model_name}"
            }

        # Make predictions
        y_pred_train = self.tuned_model.predict(X_train)
        y_pred_test = self.tuned_model.predict(X_test)

        # Calculate metrics
        if problem_type == "classification":
            metrics = {
                "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
                "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
                "train_f1": float(f1_score(y_train, y_pred_train, average='weighted', zero_division=0)),
                "test_f1": float(f1_score(y_test, y_pred_test, average='weighted', zero_division=0)),
            }
        else:
            metrics = {
                "train_mse": float(mean_squared_error(y_train, y_pred_train)),
                "test_mse": float(mean_squared_error(y_test, y_pred_test)),
                "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                "train_r2": float(r2_score(y_train, y_pred_train)),
                "test_r2": float(r2_score(y_test, y_pred_test)),
            }

        # Generate visualizations
        visualizations = self._generate_tuning_visualizations(
            search, problem_type, y_test, y_pred_test, model_name, job_id
        )

        # Get parameter importance
        param_importance = self._analyze_param_importance(search)

        return {
            "status": "success",
            "model_name": model_name,
            "best_params": self.best_params,
            "best_score": float(search.best_score_),
            "metrics": metrics,
            "predictions": {
                "train": y_pred_train.tolist()[:100],
                "test": y_pred_test.tolist()[:100]
            },
            "tuning_method": tuning_method,
            "cv_folds": cv_folds,
            "param_importance": param_importance,
            "visualizations": visualizations,
            "summary": self._generate_tuning_summary(
                model_name, self.best_params, metrics, problem_type
            )
        }

    def _get_param_grid(self, model_name: str, method: str) -> Optional[Dict[str, List]]:
        """Get hyperparameter grid for each model"""
        grids = {
            "Logistic Regression": {
                "C": [0.01, 0.1, 1, 10, 100],
                "penalty": ['l2'],
                "solver": ['lbfgs', 'liblinear']
            },
            "Decision Tree": {
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "Random Forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "Gradient Boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0]
            },
            "SVM": {
                "C": [0.1, 1, 10],
                "kernel": ['linear', 'rbf'],
                "gamma": ['scale', 'auto']
            },
            "K-Nearest Neighbors": {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ['uniform', 'distance'],
                "metric": ['euclidean', 'manhattan']
            },
            "Linear Regression": {},  # No hyperparameters to tune
            "Ridge Regression": {
                "alpha": [0.01, 0.1, 1, 10, 100]
            },
            "Lasso Regression": {
                "alpha": [0.01, 0.1, 1, 10, 100]
            },
            "SVR": {
                "C": [0.1, 1, 10],
                "kernel": ['linear', 'rbf'],
                "gamma": ['scale', 'auto']
            }
        }

        # For regressor versions, update keys
        if model_name not in grids:
            # Try to find similar model (e.g., "Decision Tree" for "DecisionTreeRegressor")
            for key in grids.keys():
                if key in model_name:
                    return grids[key]

        return grids.get(model_name, None)

    def _analyze_param_importance(self, search) -> Dict[str, Any]:
        """Analyze which parameters had the most impact"""
        if not hasattr(search, 'cv_results_'):
            return {}

        cv_results = pd.DataFrame(search.cv_results_)

        # Get parameter columns
        param_cols = [col for col in cv_results.columns if col.startswith('param_')]

        importance = {}
        for param_col in param_cols:
            param_name = param_col.replace('param_', '')
            # Group by parameter value and get mean score
            grouped = cv_results.groupby(param_col)['mean_test_score'].agg(['mean', 'std', 'count'])
            importance[param_name] = {
                "values": grouped.index.astype(str).tolist(),
                "mean_scores": grouped['mean'].tolist(),
                "std_scores": grouped['std'].tolist()
            }

        return importance

    def _generate_tuning_visualizations(
        self,
        search,
        problem_type: str,
        y_test: pd.Series,
        y_pred: np.ndarray,
        model_name: str,
        job_id: str
    ) -> List[str]:
        """Generate visualizations for hyperparameter tuning results"""
        viz_paths = []

        try:
            # 1. Parameter importance visualization
            param_importance = self._analyze_param_importance(search)

            if param_importance:
                n_params = len(param_importance)
                if n_params > 0:
                    fig, axes = plt.subplots(1, min(n_params, 3), figsize=(15, 4))
                    if n_params == 1:
                        axes = [axes]

                    for idx, (param_name, param_data) in enumerate(list(param_importance.items())[:3]):
                        ax = axes[idx] if n_params > 1 else axes[0]
                        values = param_data['values']
                        scores = param_data['mean_scores']

                        ax.plot(range(len(values)), scores, marker='o', linewidth=2, markersize=8)
                        ax.set_xticks(range(len(values)))
                        ax.set_xticklabels(values, rotation=45, ha='right')
                        ax.set_xlabel(param_name)
                        ax.set_ylabel('CV Score')
                        ax.set_title(f'Impact of {param_name}')
                        ax.grid(alpha=0.3)

                    plt.tight_layout()
                    path = self.static_dir / f"{job_id}_param_importance.png"
                    plt.savefig(path, dpi=100, bbox_inches='tight')
                    plt.close()
                    viz_paths.append(f"static/plots/{job_id}_param_importance.png")

            # 2. Prediction visualization
            if problem_type == "regression":
                plt.figure(figsize=(10, 6))
                plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                plt.title(f'Tuned Model: Actual vs Predicted - {model_name}')
                plt.grid(alpha=0.3)
                plt.tight_layout()

                path = self.static_dir / f"{job_id}_tuned_predictions.png"
                plt.savefig(path, dpi=100, bbox_inches='tight')
                plt.close()
                viz_paths.append(f"static/plots/{job_id}_tuned_predictions.png")

                # Residuals plot
                residuals = y_test - y_pred
                plt.figure(figsize=(10, 6))
                plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
                plt.axhline(y=0, color='r', linestyle='--', lw=2)
                plt.xlabel('Predicted Values')
                plt.ylabel('Residuals')
                plt.title(f'Residual Plot - {model_name}')
                plt.grid(alpha=0.3)
                plt.tight_layout()

                path = self.static_dir / f"{job_id}_residuals.png"
                plt.savefig(path, dpi=100, bbox_inches='tight')
                plt.close()
                viz_paths.append(f"static/plots/{job_id}_residuals.png")

            else:  # classification
                cm = confusion_matrix(y_test, y_pred)

                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Tuned Model: Confusion Matrix - {model_name}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()

                path = self.static_dir / f"{job_id}_tuned_confusion.png"
                plt.savefig(path, dpi=100, bbox_inches='tight')
                plt.close()
                viz_paths.append(f"static/plots/{job_id}_tuned_confusion.png")

        except Exception as e:
            print(f"Error generating tuning visualizations: {e}")

        return viz_paths

    def _generate_tuning_summary(
        self,
        model_name: str,
        best_params: Dict[str, Any],
        metrics: Dict[str, Any],
        problem_type: str
    ) -> str:
        """Generate summary of hyperparameter tuning"""
        summary_parts = [
            f"Hyperparameter Tuning completed for {model_name}",
            "",
            "Best Parameters:",
        ]

        for param, value in best_params.items():
            summary_parts.append(f"  - {param}: {value}")

        summary_parts.append("")
        summary_parts.append("Tuned Model Performance:")

        if problem_type == "classification":
            summary_parts.extend([
                f"  - Train Accuracy: {metrics['train_accuracy']:.4f}",
                f"  - Test Accuracy: {metrics['test_accuracy']:.4f}",
                f"  - Test F1 Score: {metrics['test_f1']:.4f}"
            ])
        else:
            summary_parts.extend([
                f"  - Train R²: {metrics['train_r2']:.4f}",
                f"  - Test R²: {metrics['test_r2']:.4f}",
                f"  - Test RMSE: {metrics['test_rmse']:.4f}"
            ])

        return "\n".join(summary_parts)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with tuned model"""
        if self.tuned_model is None:
            raise ValueError("No tuned model available. Run tune_and_predict first.")
        return self.tuned_model.predict(X)

    def get_model(self):
        """Return the tuned model"""
        return self.tuned_model

    def save_predictions(
        self,
        X: pd.DataFrame,
        y_true: Optional[pd.Series],
        output_path: str
    ) -> str:
        """Save predictions to CSV file"""
        predictions = self.predict(X)

        results_df = X.copy()
        results_df['prediction'] = predictions

        if y_true is not None:
            results_df['actual'] = y_true.values
            results_df['error'] = y_true.values - predictions

        results_df.to_csv(output_path, index=False)
        return output_path

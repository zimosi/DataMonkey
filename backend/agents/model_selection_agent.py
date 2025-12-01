"""
Agent 3: Model Selection & Evaluation Agent
Tries multiple off-the-shelf models and evaluates them
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


class ModelSelectionAgent:
    """
    Agent for selecting and evaluating multiple ML models
    """

    def __init__(self):
        self.static_dir = Path("backend/static/plots")
        self.static_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None

    def select_and_evaluate_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,
        test_size: float = 0.2,
        random_state: int = 42,
        cv_folds: int = 5,
        job_id: str = ""
    ) -> Dict[str, Any]:
        """
        Train and evaluate multiple models

        Args:
            X: Feature dataframe
            y: Target series
            problem_type: 'classification' or 'regression'
            test_size: Test set size
            random_state: Random state for reproducibility
            cv_folds: Number of cross-validation folds
            job_id: Job identifier for visualizations

        Returns:
            Dictionary with model results and evaluations
        """
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Get models to try
        models_to_try = self._get_models(problem_type)

        # Train and evaluate each model
        results = []
        for model_name, model in models_to_try.items():
            try:
                result = self._train_and_evaluate(
                    model, model_name, X_train, X_test, y_train, y_test,
                    problem_type, cv_folds
                )
                results.append(result)
                self.models[model_name] = model
                self.results[model_name] = result
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                results.append({
                    "model_name": model_name,
                    "status": "failed",
                    "error": str(e)
                })

        # Determine best model
        best_result = self._select_best_model(results, problem_type)
        if best_result:
            self.best_model_name = best_result["model_name"]
            self.best_model = self.models.get(self.best_model_name)

        # Generate visualizations
        visualizations = self._generate_model_visualizations(
            results, problem_type, X_test, y_test, job_id
        )

        # Generate comparison table
        comparison_df = self._create_comparison_dataframe(results, problem_type)

        return {
            "results": results,
            "best_model": best_result,
            "comparison_table": comparison_df.to_dict(),
            "visualizations": visualizations,
            "train_test_split": {
                "train_size": len(X_train),
                "test_size": len(X_test),
                "train_percentage": (1 - test_size) * 100,
                "test_percentage": test_size * 100
            },
            "models_trained": list(models_to_try.keys()),
            "problem_type": problem_type,
            "summary": self._generate_summary(results, best_result, problem_type)
        }

    def _get_models(self, problem_type: str) -> Dict[str, Any]:
        """Get list of models to try based on problem type"""
        if problem_type == "classification":
            return {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "SVM": SVC(random_state=42),
                "Naive Bayes": GaussianNB(),
                "K-Nearest Neighbors": KNeighborsClassifier()
            }
        else:  # regression
            return {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(random_state=42),
                "Lasso Regression": Lasso(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "SVR": SVR(),
                "K-Nearest Neighbors": KNeighborsRegressor()
            }

    def _train_and_evaluate(
        self,
        model,
        model_name: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        problem_type: str,
        cv_folds: int
    ) -> Dict[str, Any]:
        """Train and evaluate a single model"""
        # Train model
        model.fit(X_train, y_train)

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Cross-validation score
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())
        except:
            cv_mean = None
            cv_std = None

        # Calculate metrics based on problem type
        if problem_type == "classification":
            metrics = self._calculate_classification_metrics(
                y_train, y_pred_train, y_test, y_pred_test
            )
        else:
            metrics = self._calculate_regression_metrics(
                y_train, y_pred_train, y_test, y_pred_test
            )

        return {
            "model_name": model_name,
            "status": "success",
            "metrics": metrics,
            "cv_score_mean": cv_mean,
            "cv_score_std": cv_std,
            "predictions": {
                "train": y_pred_train.tolist()[:100],  # First 100 predictions
                "test": y_pred_test.tolist()[:100]
            }
        }

    def _calculate_classification_metrics(
        self, y_train, y_pred_train, y_test, y_pred_test
    ) -> Dict[str, Any]:
        """Calculate classification metrics"""
        # Determine average method based on binary or multiclass
        n_classes = len(np.unique(y_train))
        avg_method = 'binary' if n_classes == 2 else 'weighted'

        return {
            "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
            "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
            "train_precision": float(precision_score(y_train, y_pred_train, average=avg_method, zero_division=0)),
            "test_precision": float(precision_score(y_test, y_pred_test, average=avg_method, zero_division=0)),
            "train_recall": float(recall_score(y_train, y_pred_train, average=avg_method, zero_division=0)),
            "test_recall": float(recall_score(y_test, y_pred_test, average=avg_method, zero_division=0)),
            "train_f1": float(f1_score(y_train, y_pred_train, average=avg_method, zero_division=0)),
            "test_f1": float(f1_score(y_test, y_pred_test, average=avg_method, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist()
        }

    def _calculate_regression_metrics(
        self, y_train, y_pred_train, y_test, y_pred_test
    ) -> Dict[str, Any]:
        """Calculate regression metrics"""
        return {
            "train_mse": float(mean_squared_error(y_train, y_pred_train)),
            "test_mse": float(mean_squared_error(y_test, y_pred_test)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
            "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
            "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
            "train_r2": float(r2_score(y_train, y_pred_train)),
            "test_r2": float(r2_score(y_test, y_pred_test))
        }

    def _select_best_model(
        self, results: List[Dict[str, Any]], problem_type: str
    ) -> Optional[Dict[str, Any]]:
        """Select the best model based on performance"""
        successful_results = [r for r in results if r.get("status") == "success"]

        if not successful_results:
            return None

        if problem_type == "classification":
            # Best based on test accuracy
            best = max(successful_results, key=lambda x: x["metrics"]["test_accuracy"])
        else:
            # Best based on test R2 score
            best = max(successful_results, key=lambda x: x["metrics"]["test_r2"])

        return best

    def _create_comparison_dataframe(
        self, results: List[Dict[str, Any]], problem_type: str
    ) -> pd.DataFrame:
        """Create a comparison dataframe of model performances"""
        data = []

        for result in results:
            if result.get("status") == "success":
                row = {"Model": result["model_name"]}

                if problem_type == "classification":
                    row.update({
                        "Train Accuracy": result["metrics"]["train_accuracy"],
                        "Test Accuracy": result["metrics"]["test_accuracy"],
                        "Train F1": result["metrics"]["train_f1"],
                        "Test F1": result["metrics"]["test_f1"],
                        "CV Score": result.get("cv_score_mean", 0)
                    })
                else:
                    row.update({
                        "Train R2": result["metrics"]["train_r2"],
                        "Test R2": result["metrics"]["test_r2"],
                        "Train RMSE": result["metrics"]["train_rmse"],
                        "Test RMSE": result["metrics"]["test_rmse"],
                        "CV Score": result.get("cv_score_mean", 0)
                    })

                data.append(row)

        df = pd.DataFrame(data)

        # Sort by test performance
        if problem_type == "classification":
            df = df.sort_values("Test Accuracy", ascending=False)
        else:
            df = df.sort_values("Test R2", ascending=False)

        return df

    def _generate_model_visualizations(
        self,
        results: List[Dict[str, Any]],
        problem_type: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        job_id: str
    ) -> List[str]:
        """Generate visualizations for model comparison"""
        viz_paths = []
        successful_results = [r for r in results if r.get("status") == "success"]

        if not successful_results:
            return viz_paths

        try:
            # 1. Model comparison bar chart
            fig, ax = plt.subplots(figsize=(12, 6))

            model_names = [r["model_name"] for r in successful_results]

            if problem_type == "classification":
                train_scores = [r["metrics"]["train_accuracy"] for r in successful_results]
                test_scores = [r["metrics"]["test_accuracy"] for r in successful_results]
                ylabel = "Accuracy"
            else:
                train_scores = [r["metrics"]["train_r2"] for r in successful_results]
                test_scores = [r["metrics"]["test_r2"] for r in successful_results]
                ylabel = "R² Score"

            x = np.arange(len(model_names))
            width = 0.35

            ax.bar(x - width/2, train_scores, width, label='Train', alpha=0.8)
            ax.bar(x + width/2, test_scores, width, label='Test', alpha=0.8)

            ax.set_xlabel('Models')
            ax.set_ylabel(ylabel)
            ax.set_title(f'Model Performance Comparison ({problem_type.capitalize()})')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            path = self.static_dir / f"{job_id}_model_comparison.png"
            plt.savefig(path, dpi=100, bbox_inches='tight')
            plt.close()
            viz_paths.append(f"static/plots/{job_id}_model_comparison.png")

            # 2. Confusion matrix for best classification model
            if problem_type == "classification" and self.best_model:
                y_pred = self.best_model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)

                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - {self.best_model_name}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()

                path = self.static_dir / f"{job_id}_confusion_matrix.png"
                plt.savefig(path, dpi=100, bbox_inches='tight')
                plt.close()
                viz_paths.append(f"static/plots/{job_id}_confusion_matrix.png")

            # 3. Actual vs Predicted for best regression model
            if problem_type == "regression" and self.best_model:
                y_pred = self.best_model.predict(X_test)

                plt.figure(figsize=(10, 6))
                plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                plt.title(f'Actual vs Predicted - {self.best_model_name}')
                plt.grid(alpha=0.3)
                plt.tight_layout()

                path = self.static_dir / f"{job_id}_actual_vs_predicted.png"
                plt.savefig(path, dpi=100, bbox_inches='tight')
                plt.close()
                viz_paths.append(f"static/plots/{job_id}_actual_vs_predicted.png")

            # 4. Feature importance for tree-based models
            if self.best_model and hasattr(self.best_model, 'feature_importances_'):
                importances = self.best_model.feature_importances_
                indices = np.argsort(importances)[::-1][:15]  # Top 15 features

                plt.figure(figsize=(10, 6))
                plt.bar(range(len(indices)), importances[indices])
                plt.xticks(range(len(indices)), X_test.columns[indices], rotation=45, ha='right')
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.title(f'Feature Importance - {self.best_model_name}')
                plt.tight_layout()

                path = self.static_dir / f"{job_id}_feature_importance.png"
                plt.savefig(path, dpi=100, bbox_inches='tight')
                plt.close()
                viz_paths.append(f"static/plots/{job_id}_feature_importance.png")

        except Exception as e:
            print(f"Error generating model visualizations: {e}")

        return viz_paths

    def _generate_summary(
        self,
        results: List[Dict[str, Any]],
        best_result: Optional[Dict[str, Any]],
        problem_type: str
    ) -> str:
        """Generate text summary of model selection"""
        successful = len([r for r in results if r.get("status") == "success"])
        failed = len([r for r in results if r.get("status") == "failed"])

        summary_parts = [
            f"Model Selection completed for {problem_type} problem.",
            f"Models trained: {successful} successful, {failed} failed",
            ""
        ]

        if best_result:
            summary_parts.append(f"Best Model: {best_result['model_name']}")
            summary_parts.append("Performance:")

            if problem_type == "classification":
                summary_parts.extend([
                    f"  - Train Accuracy: {best_result['metrics']['train_accuracy']:.4f}",
                    f"  - Test Accuracy: {best_result['metrics']['test_accuracy']:.4f}",
                    f"  - Test F1 Score: {best_result['metrics']['test_f1']:.4f}"
                ])
            else:
                summary_parts.extend([
                    f"  - Train R²: {best_result['metrics']['train_r2']:.4f}",
                    f"  - Test R²: {best_result['metrics']['test_r2']:.4f}",
                    f"  - Test RMSE: {best_result['metrics']['test_rmse']:.4f}"
                ])

            if best_result.get("cv_score_mean"):
                summary_parts.append(f"  - CV Score: {best_result['cv_score_mean']:.4f} ± {best_result.get('cv_score_std', 0):.4f}")

        return "\n".join(summary_parts)

    def get_best_model(self):
        """Return the best trained model"""
        return self.best_model

    def get_model(self, model_name: str):
        """Return a specific trained model"""
        return self.models.get(model_name)

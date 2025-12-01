"""
Agent 2: Preprocessing Agent
Handles data preprocessing with configurable options
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class PreprocessingAgent:
    """
    Agent for data preprocessing operations
    """

    def __init__(self):
        self.static_dir = Path("backend/static/plots")
        self.static_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessing_history = []
        self.transformers = {}

    def preprocess_data(
        self,
        df: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
        target_column: Optional[str] = None,
        job_id: str = ""
    ) -> Dict[str, Any]:
        """
        Preprocess dataset with configurable options

        Args:
            df: Input DataFrame
            config: Preprocessing configuration
            target_column: Name of target column (will not be preprocessed)
            job_id: Job identifier for visualizations

        Returns:
            Dictionary with preprocessed data and metadata
        """
        if config is None:
            config = self._get_default_config()

        df_processed = df.copy()
        steps_performed = []
        visualizations = []

        # Separate features and target
        if target_column and target_column in df.columns:
            y = df_processed[target_column].copy()
            X = df_processed.drop(columns=[target_column])
        else:
            y = None
            X = df_processed.copy()

        original_shape = X.shape

        # 1. Handle missing values
        if config.get("handle_missing", True):
            X, missing_info = self._handle_missing_values(
                X, config.get("missing_strategy", "auto")
            )
            steps_performed.append(missing_info)

        # 2. Handle outliers
        if config.get("handle_outliers", True):
            X, outlier_info = self._handle_outliers(
                X, config.get("outlier_method", "iqr"),
                config.get("outlier_threshold", 1.5)
            )
            steps_performed.append(outlier_info)

        # 3. Handle duplicates
        if config.get("handle_duplicates", True):
            X, dup_info = self._handle_duplicates(X)
            steps_performed.append(dup_info)

        # 4. Encode categorical variables
        if config.get("encode_categorical", True):
            X, encoding_info = self._encode_categorical(
                X, config.get("encoding_method", "auto")
            )
            steps_performed.append(encoding_info)

        # 5. Scale numeric features
        if config.get("scale_features", True):
            X, scaling_info = self._scale_features(
                X, config.get("scaling_method", "standard")
            )
            steps_performed.append(scaling_info)

        # 6. Handle constant features
        if config.get("remove_constant", True):
            X, constant_info = self._remove_constant_features(X)
            steps_performed.append(constant_info)

        # 7. Handle highly correlated features
        if config.get("remove_correlated", False):
            X, corr_info = self._remove_correlated_features(
                X, config.get("correlation_threshold", 0.95)
            )
            steps_performed.append(corr_info)

        # Generate visualizations
        visualizations = self._generate_preprocessing_visualizations(
            df[[col for col in df.columns if col != target_column]], X, job_id
        )

        # Combine X and y back if target exists
        if y is not None:
            df_final = X.copy()
            df_final[target_column] = y.values
        else:
            df_final = X.copy()

        final_shape = X.shape

        return {
            "processed_dataframe": df_final,
            "feature_dataframe": X,
            "target_series": y,
            "original_shape": original_shape,
            "final_shape": final_shape,
            "steps_performed": steps_performed,
            "visualizations": visualizations,
            "config_used": config,
            "transformers": self.transformers,
            "columns_removed": list(set(df.columns) - set(df_final.columns)),
            "columns_added": list(set(df_final.columns) - set(df.columns)),
            "summary": self._generate_preprocessing_summary(
                original_shape, final_shape, steps_performed
            )
        }

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default preprocessing configuration"""
        return {
            "handle_missing": True,
            "missing_strategy": "auto",  # auto, mean, median, mode, drop
            "handle_outliers": True,
            "outlier_method": "iqr",  # iqr, zscore
            "outlier_threshold": 1.5,
            "handle_duplicates": True,
            "encode_categorical": True,
            "encoding_method": "auto",  # auto, onehot, label
            "scale_features": True,
            "scaling_method": "standard",  # standard, minmax, robust
            "remove_constant": True,
            "remove_correlated": False,
            "correlation_threshold": 0.95
        }

    def _handle_missing_values(
        self, df: pd.DataFrame, strategy: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle missing values in the dataset"""
        df_clean = df.copy()
        missing_before = df.isna().sum().sum()
        columns_affected = df.columns[df.isna().any()].tolist()

        if missing_before == 0:
            return df_clean, {
                "step": "Handle Missing Values",
                "action": "No missing values found",
                "rows_affected": 0
            }

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        # Numeric columns
        for col in numeric_cols:
            if df[col].isna().any():
                if strategy == "auto":
                    # Use median for skewed data, mean otherwise
                    if abs(df[col].skew()) > 1:
                        imputer = SimpleImputer(strategy='median')
                    else:
                        imputer = SimpleImputer(strategy='mean')
                elif strategy == "mean":
                    imputer = SimpleImputer(strategy='mean')
                elif strategy == "median":
                    imputer = SimpleImputer(strategy='median')
                else:
                    imputer = SimpleImputer(strategy='mean')

                df_clean[col] = imputer.fit_transform(df[[col]])
                self.transformers[f"imputer_{col}"] = imputer

        # Categorical columns
        for col in categorical_cols:
            if df[col].isna().any():
                if strategy == "drop":
                    df_clean = df_clean.dropna(subset=[col])
                else:
                    imputer = SimpleImputer(strategy='most_frequent')
                    df_clean[col] = imputer.fit_transform(df[[col]]).ravel()
                    self.transformers[f"imputer_{col}"] = imputer

        missing_after = df_clean.isna().sum().sum()

        return df_clean, {
            "step": "Handle Missing Values",
            "action": f"Imputed missing values using {strategy} strategy",
            "columns_affected": columns_affected,
            "missing_before": int(missing_before),
            "missing_after": int(missing_after),
            "rows_before": len(df),
            "rows_after": len(df_clean)
        }

    def _handle_outliers(
        self, df: pd.DataFrame, method: str, threshold: float
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle outliers in numeric columns"""
        df_clean = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers_removed = 0
        columns_affected = []

        for col in numeric_cols:
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                if outlier_mask.any():
                    # Cap outliers instead of removing
                    df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                    df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
                    outliers_removed += outlier_mask.sum()
                    columns_affected.append(col)

            elif method == "zscore":
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_mask = z_scores > threshold
                if outlier_mask.any():
                    # Cap outliers
                    mean = df[col].mean()
                    std = df[col].std()
                    lower_bound = mean - threshold * std
                    upper_bound = mean + threshold * std
                    df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                    df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
                    outliers_removed += outlier_mask.sum()
                    columns_affected.append(col)

        return df_clean, {
            "step": "Handle Outliers",
            "action": f"Capped outliers using {method} method (threshold={threshold})",
            "columns_affected": columns_affected,
            "outliers_capped": int(outliers_removed)
        }

    def _handle_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove duplicate rows"""
        duplicates_count = df.duplicated().sum()
        df_clean = df.drop_duplicates()

        return df_clean, {
            "step": "Handle Duplicates",
            "action": "Removed duplicate rows",
            "duplicates_removed": int(duplicates_count),
            "rows_before": len(df),
            "rows_after": len(df_clean)
        }

    def _encode_categorical(
        self, df: pd.DataFrame, method: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Encode categorical variables"""
        df_encoded = df.copy()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if len(categorical_cols) == 0:
            return df_encoded, {
                "step": "Encode Categorical",
                "action": "No categorical columns to encode",
                "columns_affected": []
            }

        columns_affected = []
        encoding_details = {}

        for col in categorical_cols:
            unique_count = df[col].nunique()

            # Decide encoding method
            if method == "auto":
                if unique_count == 2:
                    use_method = "label"
                elif unique_count <= 10:
                    use_method = "onehot"
                else:
                    use_method = "label"
            else:
                use_method = method

            # Apply encoding
            if use_method == "label":
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
                self.transformers[f"encoder_{col}"] = le
                encoding_details[col] = {
                    "method": "label",
                    "classes": le.classes_.tolist()[:10]  # First 10 classes
                }
            elif use_method == "onehot":
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df_encoded = df_encoded.drop(columns=[col])
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                encoding_details[col] = {
                    "method": "onehot",
                    "new_columns": dummies.columns.tolist()
                }

            columns_affected.append(col)

        return df_encoded, {
            "step": "Encode Categorical",
            "action": f"Encoded categorical variables using {method} method",
            "columns_affected": columns_affected,
            "encoding_details": encoding_details
        }

    def _scale_features(
        self, df: pd.DataFrame, method: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Scale numeric features"""
        df_scaled = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) == 0:
            return df_scaled, {
                "step": "Scale Features",
                "action": "No numeric columns to scale",
                "columns_affected": []
            }

        # Choose scaler
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        # Fit and transform
        df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        self.transformers["scaler"] = scaler

        return df_scaled, {
            "step": "Scale Features",
            "action": f"Scaled numeric features using {method} scaler",
            "columns_affected": numeric_cols,
            "scaler_type": method
        }

    def _remove_constant_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove features with constant values"""
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        df_clean = df.drop(columns=constant_cols)

        return df_clean, {
            "step": "Remove Constant Features",
            "action": "Removed features with only one unique value",
            "columns_removed": constant_cols,
            "count": len(constant_cols)
        }

    def _remove_correlated_features(
        self, df: pd.DataFrame, threshold: float
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove highly correlated features"""
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) < 2:
            return df, {
                "step": "Remove Correlated Features",
                "action": "Not enough numeric columns for correlation analysis",
                "columns_removed": []
            }

        corr_matrix = numeric_df.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [
            column for column in upper_tri.columns
            if any(upper_tri[column] > threshold)
        ]

        df_clean = df.drop(columns=to_drop)

        return df_clean, {
            "step": "Remove Correlated Features",
            "action": f"Removed features with correlation > {threshold}",
            "columns_removed": to_drop,
            "count": len(to_drop)
        }

    def _generate_preprocessing_visualizations(
        self, df_before: pd.DataFrame, df_after: pd.DataFrame, job_id: str
    ) -> List[str]:
        """Generate before/after visualizations"""
        viz_paths = []

        try:
            # 1. Before/After comparison for numeric columns
            numeric_cols = df_before.select_dtypes(include=[np.number]).columns[:4]
            if len(numeric_cols) > 0:
                fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(12, 4 * len(numeric_cols)))
                if len(numeric_cols) == 1:
                    axes = axes.reshape(1, -1)

                for idx, col in enumerate(numeric_cols):
                    # Before
                    if col in df_before.columns:
                        df_before[col].hist(bins=30, ax=axes[idx, 0], edgecolor='black', alpha=0.7)
                        axes[idx, 0].set_title(f'{col} - Before')
                        axes[idx, 0].set_ylabel('Frequency')

                    # After
                    if col in df_after.columns:
                        df_after[col].hist(bins=30, ax=axes[idx, 1], edgecolor='black', alpha=0.7, color='green')
                        axes[idx, 1].set_title(f'{col} - After')
                        axes[idx, 1].set_ylabel('Frequency')

                plt.tight_layout()
                path = self.static_dir / f"{job_id}_preprocessing_comparison.png"
                plt.savefig(path, dpi=100, bbox_inches='tight')
                plt.close()
                viz_paths.append(f"static/plots/{job_id}_preprocessing_comparison.png")

            # 2. Missing values comparison
            missing_before = df_before.isna().sum()
            missing_after = df_after.isna().sum()

            if missing_before.sum() > 0 or missing_after.sum() > 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

                missing_before[missing_before > 0].plot(kind='bar', ax=ax1, color='red', alpha=0.7)
                ax1.set_title('Missing Values - Before')
                ax1.set_ylabel('Count')
                ax1.tick_params(axis='x', rotation=45)

                missing_after[missing_after > 0].plot(kind='bar', ax=ax2, color='green', alpha=0.7)
                ax2.set_title('Missing Values - After')
                ax2.set_ylabel('Count')
                ax2.tick_params(axis='x', rotation=45)

                plt.tight_layout()
                path = self.static_dir / f"{job_id}_missing_comparison.png"
                plt.savefig(path, dpi=100, bbox_inches='tight')
                plt.close()
                viz_paths.append(f"static/plots/{job_id}_missing_comparison.png")

        except Exception as e:
            print(f"Error generating preprocessing visualizations: {e}")

        return viz_paths

    def _generate_preprocessing_summary(
        self, original_shape: tuple, final_shape: tuple, steps: List[Dict[str, Any]]
    ) -> str:
        """Generate a text summary of preprocessing"""
        summary_parts = [
            f"Preprocessing completed successfully.",
            f"Original shape: {original_shape}",
            f"Final shape: {final_shape}",
            f"Rows change: {original_shape[0]} → {final_shape[0]} ({final_shape[0] - original_shape[0]:+d})",
            f"Columns change: {original_shape[1]} → {final_shape[1]} ({final_shape[1] - original_shape[1]:+d})",
            "",
            "Steps performed:"
        ]

        for step in steps:
            summary_parts.append(f"  - {step['step']}: {step['action']}")

        return "\n".join(summary_parts)

"""
Agent 1: Data Understanding & Semantic Analysis
Analyzes the dataset and understands the semantic meaning of each column
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os


class DataUnderstandingAgent:
    """
    Agent for analyzing and understanding dataset semantics
    """

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.static_dir = Path("backend/static/plots")
        self.static_dir.mkdir(parents=True, exist_ok=True)

    def analyze_dataset(
        self,
        df: pd.DataFrame,
        user_prompt: str = "",
        job_id: str = ""
    ) -> Dict[str, Any]:
        """
        Comprehensive dataset analysis with semantic understanding

        Args:
            df: pandas DataFrame to analyze
            user_prompt: User's description or goal for the analysis
            job_id: Job identifier for saving visualizations

        Returns:
            Dictionary containing analysis results
        """
        # Basic dataset info
        basic_info = self._get_basic_info(df)

        # Statistical summary
        statistical_summary = self._get_statistical_summary(df)

        # Column analysis
        column_analysis = self._analyze_columns(df)

        # Data quality check
        data_quality = self._check_data_quality(df)

        # Generate visualizations
        visualizations = self._generate_visualizations(df, job_id)

        # Use LLM to understand semantic meaning
        semantic_analysis = self._semantic_analysis(
            df, user_prompt, basic_info, column_analysis
        )

        # Problem type detection
        problem_type = self._detect_problem_type(df, semantic_analysis)

        return {
            "basic_info": basic_info,
            "statistical_summary": statistical_summary,
            "column_analysis": column_analysis,
            "data_quality": data_quality,
            "semantic_analysis": semantic_analysis,
            "problem_type": problem_type,
            "visualizations": visualizations,
            "recommendations": self._generate_recommendations(
                data_quality, column_analysis, problem_type
            )
        }

    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract basic dataset information"""
        return {
            "shape": df.shape,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }

    def _get_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistical summary of the dataset"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        summary = {
            "numeric_columns": {},
            "categorical_columns": {}
        }

        # Numeric columns summary
        if numeric_cols:
            desc = df[numeric_cols].describe()
            summary["numeric_columns"] = desc.to_dict()

        # Categorical columns summary
        for col in categorical_cols:
            summary["categorical_columns"][col] = {
                "unique_values": int(df[col].nunique()),
                "most_common": df[col].value_counts().head(5).to_dict(),
                "missing_count": int(df[col].isna().sum())
            }

        return summary

    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze each column in detail"""
        column_analysis = {}

        for col in df.columns:
            col_data = df[col]
            analysis = {
                "dtype": str(col_data.dtype),
                "missing_count": int(col_data.isna().sum()),
                "missing_percentage": float(col_data.isna().sum() / len(df) * 100),
                "unique_count": int(col_data.nunique()),
                "unique_percentage": float(col_data.nunique() / len(df) * 100)
            }

            # Numeric column specifics
            if pd.api.types.is_numeric_dtype(col_data):
                analysis.update({
                    "min": float(col_data.min()) if not col_data.isna().all() else None,
                    "max": float(col_data.max()) if not col_data.isna().all() else None,
                    "mean": float(col_data.mean()) if not col_data.isna().all() else None,
                    "median": float(col_data.median()) if not col_data.isna().all() else None,
                    "std": float(col_data.std()) if not col_data.isna().all() else None,
                    "zeros_count": int((col_data == 0).sum()),
                    "negative_count": int((col_data < 0).sum()) if not col_data.isna().all() else 0,
                    "skewness": float(col_data.skew()) if not col_data.isna().all() else None,
                    "kurtosis": float(col_data.kurtosis()) if not col_data.isna().all() else None
                })

            # Categorical column specifics
            elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
                top_values = col_data.value_counts().head(10).to_dict()
                analysis.update({
                    "top_values": top_values,
                    "is_binary": col_data.nunique() == 2,
                    "avg_string_length": float(col_data.astype(str).str.len().mean()) if not col_data.isna().all() else None
                })

            column_analysis[col] = analysis

        return column_analysis

    def _check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data quality issues"""
        issues = []
        warnings = []

        # Missing values
        missing_cols = df.columns[df.isna().any()].tolist()
        if missing_cols:
            issues.append(f"Missing values found in {len(missing_cols)} columns: {', '.join(missing_cols[:5])}")

        # Duplicate rows
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            warnings.append(f"{dup_count} duplicate rows found")

        # High cardinality check
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > len(df) * 0.8:
                warnings.append(f"High cardinality in '{col}' ({df[col].nunique()} unique values)")

        # Constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        if constant_cols:
            warnings.append(f"Constant columns (may need removal): {', '.join(constant_cols)}")

        # Data type inconsistencies
        for col in df.select_dtypes(include=['object']).columns:
            try:
                pd.to_numeric(df[col])
                warnings.append(f"Column '{col}' is object but appears numeric")
            except (ValueError, TypeError):
                pass

        return {
            "issues": issues,
            "warnings": warnings,
            "quality_score": max(0, 100 - len(issues) * 20 - len(warnings) * 5)
        }

    def _semantic_analysis(
        self,
        df: pd.DataFrame,
        user_prompt: str,
        basic_info: Dict[str, Any],
        column_analysis: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Use LLM to understand semantic meaning of columns and dataset"""

        # Prepare column info for LLM
        column_info = []
        for col, analysis in column_analysis.items():
            sample_values = df[col].dropna().head(5).tolist()
            column_info.append({
                "name": col,
                "type": analysis["dtype"],
                "unique_count": analysis["unique_count"],
                "missing_percentage": analysis["missing_percentage"],
                "sample_values": [str(v) for v in sample_values]
            })

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data science expert analyzing a dataset.
Your task is to understand the semantic meaning of each column and the overall dataset purpose.
Provide insights about:
1. What each column likely represents (e.g., age, price, category, etc.)
2. The likely target variable for prediction
3. Which columns are features vs identifiers vs target
4. The domain this dataset likely belongs to (finance, healthcare, retail, etc.)
5. Suggested target column and problem type (classification/regression)

Respond in JSON format with:
{{
    "dataset_purpose": "description of what this dataset is for",
    "domain": "likely domain",
    "columns": {{
        "column_name": {{
            "semantic_meaning": "what this column represents",
            "role": "feature|target|identifier|metadata",
            "importance": "high|medium|low"
        }}
    }},
    "suggested_target": "column name",
    "suggested_problem_type": "classification|regression|clustering",
    "insights": ["list of key insights"]
}}"""),
            ("user", """Dataset Information:
- Shape: {shape}
- Columns: {num_columns}

Column Details:
{column_details}

User Context: {user_prompt}

Please analyze this dataset and provide semantic understanding.""")
        ])

        try:
            response = self.llm.invoke(
                prompt.format_messages(
                    shape=basic_info["shape"],
                    num_columns=basic_info["num_columns"],
                    column_details=json.dumps(column_info, indent=2),
                    user_prompt=user_prompt if user_prompt else "No additional context provided"
                )
            )

            # Parse JSON response
            content = response.content
            # Try to extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            semantic_info = json.loads(content)
            return semantic_info

        except Exception as e:
            return {
                "error": str(e),
                "dataset_purpose": "Unknown",
                "domain": "Unknown",
                "columns": {},
                "suggested_target": "",
                "suggested_problem_type": "",
                "insights": []
            }

    def _detect_problem_type(
        self,
        df: pd.DataFrame,
        semantic_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect the type of ML problem"""

        suggested_target = semantic_analysis.get("suggested_target", "")
        suggested_type = semantic_analysis.get("suggested_problem_type", "")

        problem_info = {
            "suggested_target_column": suggested_target,
            "problem_type": suggested_type,
            "confidence": "medium"
        }

        # Validate suggestion
        if suggested_target and suggested_target in df.columns:
            target_col = df[suggested_target]

            if pd.api.types.is_numeric_dtype(target_col):
                unique_ratio = target_col.nunique() / len(target_col)
                if unique_ratio < 0.05 or target_col.nunique() < 20:
                    problem_info["problem_type"] = "classification"
                    problem_info["num_classes"] = int(target_col.nunique())
                else:
                    problem_info["problem_type"] = "regression"
                problem_info["confidence"] = "high"
            else:
                problem_info["problem_type"] = "classification"
                problem_info["num_classes"] = int(target_col.nunique())
                problem_info["confidence"] = "high"

        return problem_info

    def _generate_visualizations(self, df: pd.DataFrame, job_id: str) -> List[str]:
        """Generate visualizations for data understanding"""
        viz_paths = []

        try:
            # 1. Missing values heatmap
            if df.isna().any().any():
                plt.figure(figsize=(12, 6))
                sns.heatmap(df.isna(), cbar=True, yticklabels=False, cmap='viridis')
                plt.title('Missing Values Heatmap')
                plt.tight_layout()
                path = self.static_dir / f"{job_id}_missing_values.png"
                plt.savefig(path, dpi=100, bbox_inches='tight')
                plt.close()
                viz_paths.append(f"static/plots/{job_id}_missing_values.png")

            # 2. Distribution of numeric columns (first 6)
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
            if len(numeric_cols) > 0:
                n_cols = min(3, len(numeric_cols))
                n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
                if n_rows == 1 and n_cols == 1:
                    axes = np.array([axes])
                axes = axes.flatten() if len(numeric_cols) > 1 else [axes]

                for idx, col in enumerate(numeric_cols):
                    df[col].hist(bins=30, ax=axes[idx], edgecolor='black')
                    axes[idx].set_title(f'Distribution of {col}')
                    axes[idx].set_xlabel(col)
                    axes[idx].set_ylabel('Frequency')

                # Hide extra subplots
                for idx in range(len(numeric_cols), len(axes)):
                    axes[idx].axis('off')

                plt.tight_layout()
                path = self.static_dir / f"{job_id}_distributions.png"
                plt.savefig(path, dpi=100, bbox_inches='tight')
                plt.close()
                viz_paths.append(f"static/plots/{job_id}_distributions.png")

            # 3. Correlation heatmap for numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                plt.figure(figsize=(10, 8))
                corr = numeric_df.corr()
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
                plt.title('Correlation Heatmap')
                plt.tight_layout()
                path = self.static_dir / f"{job_id}_correlation.png"
                plt.savefig(path, dpi=100, bbox_inches='tight')
                plt.close()
                viz_paths.append(f"static/plots/{job_id}_correlation.png")

            # 4. Categorical columns distribution (first 4)
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns[:4]
            if len(categorical_cols) > 0:
                n_cols = min(2, len(categorical_cols))
                n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
                if n_rows == 1 and n_cols == 1:
                    axes = np.array([axes])
                axes = axes.flatten() if len(categorical_cols) > 1 else [axes]

                for idx, col in enumerate(categorical_cols):
                    top_10 = df[col].value_counts().head(10)
                    top_10.plot(kind='bar', ax=axes[idx])
                    axes[idx].set_title(f'Top 10 values in {col}')
                    axes[idx].set_xlabel(col)
                    axes[idx].set_ylabel('Count')
                    axes[idx].tick_params(axis='x', rotation=45)

                # Hide extra subplots
                for idx in range(len(categorical_cols), len(axes)):
                    axes[idx].axis('off')

                plt.tight_layout()
                path = self.static_dir / f"{job_id}_categorical.png"
                plt.savefig(path, dpi=100, bbox_inches='tight')
                plt.close()
                viz_paths.append(f"static/plots/{job_id}_categorical.png")

        except Exception as e:
            print(f"Error generating visualizations: {e}")

        return viz_paths

    def _generate_recommendations(
        self,
        data_quality: Dict[str, Any],
        column_analysis: Dict[str, Dict[str, Any]],
        problem_type: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for next steps"""
        recommendations = []

        # Missing value recommendations
        high_missing_cols = [
            col for col, analysis in column_analysis.items()
            if analysis["missing_percentage"] > 30
        ]
        if high_missing_cols:
            recommendations.append(
                f"Consider dropping columns with high missing values: {', '.join(high_missing_cols)}"
            )

        # Feature engineering recommendations
        date_like_cols = [
            col for col in column_analysis.keys()
            if 'date' in col.lower() or 'time' in col.lower()
        ]
        if date_like_cols:
            recommendations.append(
                f"Extract datetime features from: {', '.join(date_like_cols)}"
            )

        # Encoding recommendations
        categorical_cols = [
            col for col, analysis in column_analysis.items()
            if analysis["dtype"] in ['object', 'category']
        ]
        if categorical_cols:
            recommendations.append(
                f"Encode categorical variables: {', '.join(categorical_cols[:5])}"
            )

        # Scaling recommendation
        numeric_cols = [
            col for col, analysis in column_analysis.items()
            if 'mean' in analysis
        ]
        if numeric_cols:
            recommendations.append("Consider scaling numeric features for better model performance")

        # Problem-specific recommendations
        if problem_type.get("problem_type") == "classification":
            recommendations.append("Use classification algorithms (Logistic Regression, Random Forest, XGBoost)")
        elif problem_type.get("problem_type") == "regression":
            recommendations.append("Use regression algorithms (Linear Regression, Random Forest, XGBoost)")

        return recommendations

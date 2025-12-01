"""
Intelligent Preprocessing Agent
Dynamically chooses preprocessing steps based on data characteristics using LLM
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
import sys

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from agents.adaptive_preprocessing_agent import AdaptivePreprocessingAgent
from config import get_openai_config


class IntelligentPreprocessingAgent(AdaptivePreprocessingAgent):
    """
    Intelligent preprocessing agent that uses LLM to decide which preprocessing
    steps to apply based on data characteristics
    """

    def __init__(self):
        super().__init__()
        config = get_openai_config()
        self.llm = ChatOpenAI(
            model=config['model'],
            temperature=0.3,  # Lower temperature for more deterministic preprocessing decisions
            api_key=config['api_key']
        )

    def preprocess_data(
        self,
        df: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
        target_column: Optional[str] = None,
        job_id: str = "",
        enable_sub_agents: bool = True
    ) -> Dict[str, Any]:
        """
        Intelligently preprocess data by dynamically choosing steps with LLM

        Args:
            df: Input DataFrame
            config: Optional user-provided config (will be augmented by LLM decisions)
            target_column: Name of target column
            job_id: Job identifier
            enable_sub_agents: Whether to enable sub-agent spawning

        Returns:
            Enhanced preprocessing results with LLM reasoning
        """
        # Analyze the dataset
        data_analysis = self._analyze_dataset(df, target_column)

        # Get LLM recommendations for preprocessing
        preprocessing_plan = self._get_llm_preprocessing_plan(data_analysis, config)

        # Merge LLM plan with user config (user config takes precedence)
        final_config = preprocessing_plan.copy()
        if config:
            final_config.update(config)

        # Execute preprocessing with the intelligent config
        result = super().preprocess_data(
            df=df,
            config=final_config,
            target_column=target_column,
            job_id=job_id,
            enable_sub_agents=enable_sub_agents
        )

        # Add LLM reasoning to results
        result["llm_analysis"] = data_analysis
        result["llm_preprocessing_plan"] = preprocessing_plan
        result["llm_reasoning"] = preprocessing_plan.get("reasoning", "")

        return result

    def _analyze_dataset(self, df: pd.DataFrame, target_column: Optional[str]) -> Dict[str, Any]:
        """Analyze dataset characteristics"""
        analysis = {
            "shape": df.shape,
            "total_cells": df.shape[0] * df.shape[1],
            "dtypes_summary": {},
            "missing_summary": {},
            "outlier_summary": {},
            "categorical_summary": {},
            "target_info": {}
        }

        # Dtype analysis
        for dtype in df.dtypes.unique():
            count = (df.dtypes == dtype).sum()
            analysis["dtypes_summary"][str(dtype)] = int(count)

        # Missing values
        total_missing = df.isna().sum().sum()
        analysis["missing_summary"] = {
            "total_missing": int(total_missing),
            "percentage": float(total_missing / analysis["total_cells"] * 100),
            "columns_with_missing": df.columns[df.isna().any()].tolist()
        }

        # Outlier detection (IQR method)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_columns = []
        total_outliers = 0

        for col in numeric_cols:
            if col != target_column:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                if outliers > 0:
                    outlier_columns.append({
                        "column": col,
                        "count": int(outliers),
                        "percentage": float(outliers / len(df) * 100)
                    })
                    total_outliers += outliers

        analysis["outlier_summary"] = {
            "total_outliers": int(total_outliers),
            "affected_columns": outlier_columns
        }

        # Categorical analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        cat_info = []
        for col in categorical_cols:
            if col != target_column:
                cat_info.append({
                    "column": col,
                    "unique_values": int(df[col].nunique()),
                    "cardinality": "high" if df[col].nunique() > 50 else "low"
                })

        analysis["categorical_summary"] = {
            "count": len(categorical_cols),
            "columns": cat_info[:5]  # Top 5
        }

        # Target column info
        if target_column and target_column in df.columns:
            is_numeric = pd.api.types.is_numeric_dtype(df[target_column])
            analysis["target_info"] = {
                "name": target_column,
                "type": "numeric" if is_numeric else "categorical",
                "unique_values": int(df[target_column].nunique()),
                "missing": int(df[target_column].isna().sum())
            }

        return analysis

    def _get_llm_preprocessing_plan(
        self,
        data_analysis: Dict[str, Any],
        user_config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Use LLM to decide which preprocessing steps to apply"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert data preprocessing strategist. Based on the dataset analysis,
            recommend the optimal preprocessing steps. Be specific and justify each decision.

            Return your recommendations in this JSON format:
            {{
                "handle_missing": true/false,
                "missing_strategy": "mean"/"median"/"mode"/"drop",
                "handle_outliers": true/false,
                "outlier_method": "iqr"/"zscore",
                "outlier_threshold": float,
                "handle_duplicates": true/false,
                "encode_categorical": true/false,
                "encoding_method": "auto"/"onehot"/"label",
                "scale_features": true/false,
                "scaling_method": "standard"/"minmax"/"robust",
                "remove_constant": true/false,
                "remove_correlated": true/false,
                "correlation_threshold": float,
                "reasoning": "explanation of your choices"
            }}"""),
            ("user", """Dataset Analysis:

Shape: {shape}
Total Cells: {total_cells}

Data Types:
{dtypes}

Missing Values:
- Total: {missing_total} ({missing_pct:.2f}%)
- Columns affected: {missing_cols}

Outliers:
- Total: {outlier_total}
- Affected columns: {outlier_info}

Categorical Variables:
- Count: {cat_count}
- Details: {cat_info}

Target Column:
{target_info}

User Preferences (if any):
{user_config}

Based on this analysis, what preprocessing steps should I apply and why?
Be practical and consider the trade-offs. Focus on steps that will genuinely improve model performance.""")
        ])

        # Format the data for the prompt
        messages = prompt.format_messages(
            shape=data_analysis["shape"],
            total_cells=data_analysis["total_cells"],
            dtypes="\n".join([f"  {k}: {v} columns" for k, v in data_analysis["dtypes_summary"].items()]),
            missing_total=data_analysis["missing_summary"]["total_missing"],
            missing_pct=data_analysis["missing_summary"]["percentage"],
            missing_cols=", ".join(data_analysis["missing_summary"]["columns_with_missing"][:5]),
            outlier_total=data_analysis["outlier_summary"]["total_outliers"],
            outlier_info="\n".join([
                f"  {item['column']}: {item['count']} ({item['percentage']:.1f}%)"
                for item in data_analysis["outlier_summary"]["affected_columns"][:5]
            ]),
            cat_count=data_analysis["categorical_summary"]["count"],
            cat_info="\n".join([
                f"  {item['column']}: {item['unique_values']} unique ({item['cardinality']} cardinality)"
                for item in data_analysis["categorical_summary"]["columns"]
            ]),
            target_info=str(data_analysis.get("target_info", "Not specified")),
            user_config=str(user_config) if user_config else "None - you have full discretion"
        )

        try:
            response = self.llm.invoke(messages)
            response_text = response.content

            # Extract JSON from response
            import json
            import re

            # Try to find JSON in the response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                plan = json.loads(json_match.group())
            else:
                # Fallback to safe defaults
                plan = self._get_default_config()
                plan["reasoning"] = "Failed to parse LLM response, using defaults"

            return plan

        except Exception as e:
            print(f"LLM preprocessing planning failed: {e}")
            # Fallback to intelligent defaults based on analysis
            return self._get_smart_defaults(data_analysis)

    def _get_smart_defaults(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get smart defaults based on data analysis"""
        config = {
            "handle_missing": analysis["missing_summary"]["percentage"] > 0,
            "missing_strategy": "median" if analysis["missing_summary"]["percentage"] < 30 else "drop",
            "handle_outliers": len(analysis["outlier_summary"]["affected_columns"]) > 0,
            "outlier_method": "iqr",
            "outlier_threshold": 1.5,
            "handle_duplicates": True,
            "encode_categorical": analysis["categorical_summary"]["count"] > 0,
            "encoding_method": "auto",
            "scale_features": True,
            "scaling_method": "standard",
            "remove_constant": True,
            "remove_correlated": False,
            "correlation_threshold": 0.95,
            "reasoning": "Auto-generated based on data characteristics"
        }
        return config

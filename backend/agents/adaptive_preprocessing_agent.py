"""
Adaptive Preprocessing Agent with Sub-Agent Spawning
Extends PreprocessingAgent with ability to spawn specialized sub-agents
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from agents.preprocessing_agent import PreprocessingAgent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import get_openai_config


class OutlierAnalysisSubAgent:
    """
    Specialized sub-agent for deep outlier analysis and recommendations
    """

    def __init__(self):
        config = get_openai_config()
        self.llm = ChatOpenAI(
            model=config['model'],
            temperature=config['temperature'],
            api_key=config['api_key']
        )

    def analyze_and_recommend(self, df: pd.DataFrame, outlier_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze outliers and provide intelligent recommendations

        Args:
            df: DataFrame with potential outliers
            outlier_stats: Statistics about outliers detected

        Returns:
            Dictionary with analysis and recommendations
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Calculate detailed outlier statistics
        detailed_stats = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(df)) * 100

            if outlier_count > 0:
                detailed_stats[col] = {
                    "count": outlier_count,
                    "percentage": outlier_percentage,
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "min_outlier": float(df[col].min()),
                    "max_outlier": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std())
                }

        # Use LLM to generate intelligent recommendations
        if detailed_stats:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert data scientist specializing in outlier detection and handling.
                Analyze the outlier statistics and provide specific recommendations for handling them.
                Consider the domain, data distribution, and business context."""),
                ("user", """Outlier Analysis Results:

{stats}

Based on these statistics, provide:
1. Assessment of outlier severity
2. Recommended handling strategy (cap, remove, transform, keep)
3. Reasoning for your recommendation
4. Potential risks if not handled properly

Be concise and actionable.""")
            ])

            stats_str = "\n".join([
                f"{col}: {stats['count']} outliers ({stats['percentage']:.2f}%), "
                f"bounds=[{stats['lower_bound']:.2f}, {stats['upper_bound']:.2f}]"
                for col, stats in detailed_stats.items()
            ])

            try:
                response = self.llm.invoke(prompt.format_messages(stats=stats_str))
                llm_recommendation = response.content
            except Exception as e:
                llm_recommendation = f"LLM analysis unavailable: {str(e)}"
        else:
            llm_recommendation = "No significant outliers detected."

        return {
            "sub_agent": "OutlierAnalysisSubAgent",
            "detailed_statistics": detailed_stats,
            "columns_analyzed": list(detailed_stats.keys()),
            "total_outlier_columns": len(detailed_stats),
            "llm_recommendation": llm_recommendation,
            "suggested_action": self._determine_action(detailed_stats),
            "severity": self._assess_severity(detailed_stats)
        }

    def _determine_action(self, stats: Dict[str, Any]) -> str:
        """Determine recommended action based on outlier statistics"""
        if not stats:
            return "no_action"

        avg_percentage = np.mean([s['percentage'] for s in stats.values()])

        if avg_percentage > 20:
            return "investigate_data_quality"
        elif avg_percentage > 10:
            return "transform_or_cap"
        elif avg_percentage > 5:
            return "cap_outliers"
        else:
            return "monitor_only"

    def _assess_severity(self, stats: Dict[str, Any]) -> str:
        """Assess overall severity of outlier situation"""
        if not stats:
            return "none"

        avg_percentage = np.mean([s['percentage'] for s in stats.values()])

        if avg_percentage > 15:
            return "high"
        elif avg_percentage > 5:
            return "medium"
        else:
            return "low"


class FeatureEngineeringSubAgent:
    """
    Sub-agent for creating derived features based on data insights
    """

    def __init__(self):
        config = get_openai_config()
        self.llm = ChatOpenAI(
            model=config['model'],
            temperature=config['temperature'],
            api_key=config['api_key']
        )

    def suggest_features(self, df: pd.DataFrame, data_insights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze data and suggest potential feature engineering opportunities

        Args:
            df: Input DataFrame
            data_insights: Insights from data understanding stage

        Returns:
            Dictionary with feature engineering suggestions
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        suggestions = {
            "polynomial_features": [],
            "interaction_features": [],
            "binning_features": [],
            "log_transform": [],
            "reasoning": ""
        }

        # Identify skewed features for log transform
        for col in numeric_cols:
            skewness = abs(df[col].skew())
            if skewness > 1:
                suggestions["log_transform"].append({
                    "column": col,
                    "skewness": float(skewness),
                    "reason": "High skewness detected"
                })

        # Suggest interaction features for numeric columns
        if len(numeric_cols) >= 2:
            suggestions["interaction_features"] = [
                {"columns": [numeric_cols[0], numeric_cols[1]],
                 "operation": "multiply",
                 "reason": "Potential interaction effect"}
            ]

        # Use LLM for intelligent feature suggestions
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a feature engineering expert. Suggest creative and useful
            feature transformations based on the data characteristics."""),
            ("user", """Data has:
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}
- Sample statistics: {stats}

Suggest 2-3 specific feature engineering techniques that would be most beneficial.""")
        ])

        stats_summary = {
            col: {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            }
            for col in numeric_cols[:3]  # First 3 columns
        }

        try:
            response = self.llm.invoke(prompt.format_messages(
                numeric_cols=numeric_cols[:5],
                categorical_cols=categorical_cols[:5],
                stats=stats_summary
            ))
            suggestions["reasoning"] = response.content
        except Exception as e:
            suggestions["reasoning"] = f"LLM suggestions unavailable: {str(e)}"

        return {
            "sub_agent": "FeatureEngineeringSubAgent",
            "suggestions": suggestions,
            "potential_impact": "Can improve model performance by 5-15%"
        }


class AdaptivePreprocessingAgent(PreprocessingAgent):
    """
    Adaptive preprocessing agent that can spawn specialized sub-agents
    based on data characteristics and preprocessing results
    """

    def __init__(self):
        super().__init__()
        self.sub_agents_spawned = []
        self.sub_agent_results = []

    def preprocess_data(
        self,
        df: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
        target_column: Optional[str] = None,
        job_id: str = "",
        enable_sub_agents: bool = True
    ) -> Dict[str, Any]:
        """
        Adaptive preprocessing with sub-agent spawning capability

        Args:
            df: Input DataFrame
            config: Preprocessing configuration
            target_column: Name of target column
            job_id: Job identifier
            enable_sub_agents: Whether to enable sub-agent spawning

        Returns:
            Enhanced preprocessing results with sub-agent insights
        """
        # Run standard preprocessing
        results = super().preprocess_data(df, config, target_column, job_id)

        if not enable_sub_agents:
            return results

        # Analyze preprocessing results to decide on sub-agents
        insights = self._analyze_preprocessing_results(results)

        # Decision logic for spawning sub-agents
        sub_agent_outputs = []

        # 1. Spawn OutlierAnalysisSubAgent if outliers > 10%
        if insights.get("outlier_percentage", 0) > 10:
            print(f"[AdaptiveAgent] Spawning OutlierAnalysisSubAgent (outlier%: {insights['outlier_percentage']:.2f})")
            outlier_agent = OutlierAnalysisSubAgent()
            outlier_analysis = outlier_agent.analyze_and_recommend(
                results["feature_dataframe"],
                insights.get("outlier_stats", {})
            )
            sub_agent_outputs.append(outlier_analysis)
            self.sub_agents_spawned.append("OutlierAnalysisSubAgent")

            # Apply recommendations if severity is high
            if outlier_analysis.get("severity") == "high":
                results["warnings"] = results.get("warnings", [])
                results["warnings"].append(
                    "HIGH SEVERITY OUTLIERS DETECTED - Review OutlierAnalysisSubAgent recommendations"
                )

        # 2. Spawn FeatureEngineeringSubAgent if complexity is low
        if insights.get("feature_count", 0) < 10:
            print(f"[AdaptiveAgent] Spawning FeatureEngineeringSubAgent (feature count: {insights['feature_count']})")
            fe_agent = FeatureEngineeringSubAgent()
            fe_suggestions = fe_agent.suggest_features(
                results["feature_dataframe"],
                insights
            )
            sub_agent_outputs.append(fe_suggestions)
            self.sub_agents_spawned.append("FeatureEngineeringSubAgent")

        # Enhance results with sub-agent outputs
        results["sub_agents"] = {
            "spawned": self.sub_agents_spawned,
            "count": len(self.sub_agents_spawned),
            "outputs": sub_agent_outputs
        }

        results["adaptive_insights"] = insights
        results["summary"] += f"\n\nAdaptive Analysis: Spawned {len(self.sub_agents_spawned)} sub-agent(s)"

        return results

    def _analyze_preprocessing_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze preprocessing results to determine if sub-agents needed

        Args:
            results: Preprocessing results from parent class

        Returns:
            Dictionary with insights for decision making
        """
        insights = {}

        # Analyze outlier situation
        outlier_step = next(
            (step for step in results["steps_performed"] if step["step"] == "Handle Outliers"),
            None
        )

        if outlier_step:
            outliers_capped = outlier_step.get("outliers_capped", 0)
            total_cells = results["original_shape"][0] * len(outlier_step.get("columns_affected", []))
            outlier_percentage = (outliers_capped / total_cells * 100) if total_cells > 0 else 0

            insights["outlier_percentage"] = outlier_percentage
            insights["outlier_stats"] = {
                "capped": outliers_capped,
                "columns": outlier_step.get("columns_affected", [])
            }

        # Feature complexity
        insights["feature_count"] = results["final_shape"][1]
        insights["row_count"] = results["final_shape"][0]

        # Missing data complexity
        missing_step = next(
            (step for step in results["steps_performed"] if step["step"] == "Handle Missing Values"),
            None
        )

        if missing_step:
            missing_percentage = (
                missing_step.get("missing_before", 0) /
                (results["original_shape"][0] * results["original_shape"][1]) * 100
            ) if results["original_shape"][0] > 0 else 0
            insights["missing_percentage"] = missing_percentage

        return insights

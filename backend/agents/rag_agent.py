"""
RAG Agent - Retrieval Augmented Generation for Preprocessing Recommendations
"""
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from knowledge_base.preprocessing_knowledge import get_recommendations, PREPROCESSING_KNOWLEDGE


class RAGPreprocessingAgent:
    """
    RAG-powered agent that provides preprocessing recommendations
    """

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.knowledge_base = PREPROCESSING_KNOWLEDGE

    def get_recommendations(self, data_stats: Dict[str, Any], user_context: str = "") -> Dict[str, Any]:
        """
        Get preprocessing recommendations using RAG

        Args:
            data_stats: Statistics about the dataset
            user_context: Additional context from user

        Returns:
            Dictionary with recommendations and explanations
        """
        # Get structured recommendations from knowledge base
        kb_recommendations = get_recommendations(data_stats)

        # Use LLM to synthesize and explain
        explanation = self._generate_explanation(data_stats, kb_recommendations, user_context)

        return {
            "recommendations": kb_recommendations,
            "explanation": explanation,
            "data_stats": data_stats
        }

    def _generate_explanation(
        self,
        data_stats: Dict[str, Any],
        recommendations: Dict[str, Any],
        user_context: str
    ) -> str:
        """Generate natural language explanation of recommendations"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data preprocessing expert. Explain preprocessing recommendations
in a clear, educational way. Focus on the 'why' behind each recommendation."""),
            ("user", """Dataset Statistics:
- Rows: {num_rows}
- Columns: {num_cols}
- Missing values: {missing_pct:.1f}%
- Outliers: {outlier_pct:.1f}%
- Categorical columns: {num_cat_cols}

Recommendations from knowledge base:
{recommendations}

User context: {user_context}

Provide a concise explanation (3-5 sentences) of why these preprocessing steps are recommended for this specific dataset.""")
        ])

        try:
            response = self.llm.invoke(
                prompt.format_messages(
                    num_rows=data_stats.get('num_rows', 0),
                    num_cols=data_stats.get('num_cols', 0),
                    missing_pct=data_stats.get('missing_percentage', 0),
                    outlier_pct=data_stats.get('outlier_percentage', 0),
                    num_cat_cols=data_stats.get('num_categorical_cols', 0),
                    recommendations=str(recommendations),
                    user_context=user_context or "No additional context"
                )
            )
            return response.content
        except Exception as e:
            return f"Could not generate explanation: {str(e)}"

    def explain_technique(self, technique: str, category: str) -> Dict[str, Any]:
        """
        Explain a specific preprocessing technique

        Args:
            technique: Name of the technique (e.g., 'mean_imputation')
            category: Category (e.g., 'missing_values', 'outliers')

        Returns:
            Dictionary with technique details
        """
        try:
            if category in self.knowledge_base:
                # Navigate knowledge base
                cat_data = self.knowledge_base[category]

                # Handle nested structure
                for key, value in cat_data.items():
                    if isinstance(value, dict) and technique in value:
                        return value[technique]
                    elif key == technique:
                        return value

            return {"error": f"Technique '{technique}' not found in category '{category}'"}
        except Exception as e:
            return {"error": str(e)}

    def suggest_next_steps(self, current_results: Dict[str, Any]) -> List[str]:
        """
        Suggest next preprocessing steps based on current results

        Args:
            current_results: Results from current preprocessing

        Returns:
            List of suggested next steps
        """
        suggestions = []

        # Check if further preprocessing needed
        if current_results.get('remaining_missing', 0) > 0:
            suggestions.append("Some missing values remain - consider KNN imputation for better accuracy")

        if current_results.get('skewness', 0) > 2:
            suggestions.append("Data is highly skewed - consider log transformation or Box-Cox")

        if current_results.get('high_cardinality_features'):
            suggestions.append(f"High cardinality features detected - consider target encoding for: {current_results['high_cardinality_features']}")

        if current_results.get('correlated_features'):
            suggestions.append(f"Highly correlated features - consider removing redundant features: {current_results['correlated_features']}")

        if not suggestions:
            suggestions.append("Preprocessing looks good! Ready for model selection.")

        return suggestions

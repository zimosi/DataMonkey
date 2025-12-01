"""
Pipeline Orchestrator
Manages the execution of the 5-stage Data Monkey pipeline
"""
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from config import get_openai_config
from agents.data_understanding_agent import DataUnderstandingAgent
from agents.intelligent_preprocessing_agent import IntelligentPreprocessingAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.hyperparameter_agent import HyperparameterTuningAgent
from agents.rag_agent import RAGPreprocessingAgent
from knowledge_base.preprocessing_knowledge import get_recommendations

from .pipeline_state import (
    PipelineState,
    PipelineStage,
    StageStatus,
    create_initial_state,
    update_stage_status
)
from .dag_manager import DAGManager


class PipelineOrchestrator:
    """
    Orchestrates the complete Data Monkey pipeline
    """

    def __init__(self, job_id: str, dataset_path: str):
        self.job_id = job_id
        self.dataset_path = dataset_path
        self.state: PipelineState = create_initial_state(job_id, dataset_path)

        # Initialize LLM
        config = get_openai_config()
        self.llm = ChatOpenAI(
            model=config["model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"]
        )

        # Initialize agents (use IntelligentPreprocessingAgent - dynamically chooses preprocessing with LLM)
        self.data_understanding_agent = DataUnderstandingAgent(self.llm)
        self.preprocessing_agent = IntelligentPreprocessingAgent()  # MVE: Intelligent + Adaptive agent
        self.rag_agent = RAGPreprocessingAgent(self.llm)  # MVE: RAG agent (needs llm)
        self.model_selection_agent = ModelSelectionAgent()
        self.hyperparameter_agent = HyperparameterTuningAgent()

        # Initialize DAG manager
        self.dag_manager = DAGManager(job_id)

        # Load dataset
        self.df = pd.read_csv(dataset_path)
        self.processed_df = None
        self.X = None
        self.y = None

    def execute_pipeline(
        self,
        user_prompt: str = "",
        preprocessing_config: Optional[Dict[str, Any]] = None,
        target_column: Optional[str] = None,
        auto_mode: bool = True
    ) -> PipelineState:
        """
        Execute the complete pipeline

        Args:
            user_prompt: User's description or goal
            preprocessing_config: Custom preprocessing configuration
            target_column: Target column name (if known)
            auto_mode: If True, automatically detect settings

        Returns:
            Updated pipeline state
        """
        try:
            # Stage 1: Data Understanding
            self.execute_data_understanding(user_prompt)

            # Auto-detect target column if not provided
            if auto_mode and not target_column:
                target_column = self.state.get("data_understanding", {}).get("data", {}).get(
                    "problem_type", {}
                ).get("suggested_target_column", "")

            self.state["target_column"] = target_column

            # Stage 2: Preprocessing
            self.execute_preprocessing(preprocessing_config, target_column)

            # Determine problem type
            problem_type = self.state.get("data_understanding", {}).get("data", {}).get(
                "problem_type", {}
            ).get("problem_type", "classification")
            self.state["problem_type"] = problem_type

            # Stage 3: Model Selection
            self.execute_model_selection(problem_type)

            # Stage 4: Hyperparameter Tuning
            self.execute_hyperparameter_tuning(problem_type)

            return self.state

        except Exception as e:
            self.state["error"] = str(e)
            return self.state

    def execute_data_understanding(self, user_prompt: str = "") -> Dict[str, Any]:
        """Execute Stage 1: Data Understanding"""
        update_stage_status(
            self.state,
            PipelineStage.DATA_UNDERSTANDING,
            StageStatus.IN_PROGRESS,
            logs=["Starting data understanding and semantic analysis..."]
        )

        # Update DAG
        self.dag_manager.update_node_status("data_understanding", "running")

        try:
            result = self.data_understanding_agent.analyze_dataset(
                self.df,
                user_prompt=user_prompt,
                job_id=self.job_id
            )

            # Extract feature columns
            all_cols = self.df.columns.tolist()
            target = result.get("problem_type", {}).get("suggested_target_column", "")
            feature_cols = [col for col in all_cols if col != target]
            self.state["feature_columns"] = feature_cols

            # Update DAG
            self.dag_manager.update_node_status("data_understanding", "completed")

            update_stage_status(
                self.state,
                PipelineStage.DATA_UNDERSTANDING,
                StageStatus.COMPLETED,
                data=result,
                visualizations=result.get("visualizations", []),
                metrics={
                    "num_rows": result["basic_info"]["num_rows"],
                    "num_columns": result["basic_info"]["num_columns"],
                    "quality_score": result["data_quality"]["quality_score"]
                },
                logs=["Data understanding completed successfully"]
            )

            return result

        except Exception as e:
            self.dag_manager.update_node_status("data_understanding", "failed")
            update_stage_status(
                self.state,
                PipelineStage.DATA_UNDERSTANDING,
                StageStatus.FAILED,
                error=str(e),
                logs=[f"Error in data understanding: {str(e)}"]
            )
            raise

    def execute_preprocessing(
        self,
        config: Optional[Dict[str, Any]] = None,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute Stage 2: Preprocessing with RAG and Adaptive Agents"""
        update_stage_status(
            self.state,
            PipelineStage.PREPROCESSING,
            StageStatus.IN_PROGRESS,
            logs=["Starting data preprocessing..."]
        )

        # Update DAG
        self.dag_manager.update_node_status("preprocessing", "running")

        try:
            # MVE: Get RAG recommendations before preprocessing
            # Convert dtypes to serializable format
            dtypes_dict = {}
            for dtype, count in self.df.dtypes.value_counts().items():
                dtypes_dict[str(dtype)] = int(count)

            data_stats = {
                "shape": self.df.shape,
                "dtypes": dtypes_dict,
                "missing_percentage": float(self.df.isna().sum().sum() / (self.df.shape[0] * self.df.shape[1]) * 100),
                "numeric_columns": len(self.df.select_dtypes(include=['number']).columns),
                "categorical_columns": len(self.df.select_dtypes(include=['object', 'category']).columns)
            }

            # Add RAG agent to DAG
            self.dag_manager.add_decision_agent(
                agent_id="preprocessing_rag",
                parent_id="preprocessing",
                label="RAG Knowledge Agent",
                reason="Provides preprocessing best practices from knowledge base",
                agent_type="rag",
                status="running"
            )

            rag_recommendations = self.rag_agent.get_recommendations(
                data_stats,
                user_context="Preparing data for machine learning"
            )

            self.dag_manager.update_node_status("preprocessing_rag", "completed")

            # Add LLM Planning agent to DAG
            self.dag_manager.add_decision_agent(
                agent_id="preprocessing_llm_planner",
                parent_id="preprocessing",
                label="LLM Strategy Planner",
                reason="Decides which preprocessing steps to apply based on data analysis",
                agent_type="decision",
                status="running"
            )

            # MVE: Execute intelligent preprocessing (LLM decides steps, can spawn sub-agents)
            result = self.preprocessing_agent.preprocess_data(
                self.df,
                config=config,
                target_column=target_column,
                job_id=self.job_id,
                enable_sub_agents=True  # MVE: Enable sub-agent spawning
            )

            # Mark LLM planner as completed
            self.dag_manager.update_node_status("preprocessing_llm_planner", "completed")

            # MVE: Add RAG recommendations to result
            result["rag_recommendations"] = rag_recommendations

            # MVE: Update DAG with spawned sub-agents
            if "sub_agents" in result and result["sub_agents"]["spawned"]:
                for sub_agent_name in result["sub_agents"]["spawned"]:
                    sub_agent_id = f"preprocessing_{sub_agent_name.lower()}"
                    self.dag_manager.add_sub_agent(
                        sub_agent_id=sub_agent_id,
                        parent_id="preprocessing",
                        label=sub_agent_name,
                        reason=f"Spawned for specialized {sub_agent_name.replace('SubAgent', '')} analysis",
                        status="completed"
                    )

            # Store processed data
            self.processed_df = result["processed_dataframe"]
            self.X = result["feature_dataframe"]
            self.y = result["target_series"]

            # Update state
            self.state["original_data_shape"] = result["original_shape"]
            self.state["preprocessed_data_shape"] = result["final_shape"]

            # Update DAG
            self.dag_manager.update_node_status("preprocessing", "completed")

            update_stage_status(
                self.state,
                PipelineStage.PREPROCESSING,
                StageStatus.COMPLETED,
                data=result,
                visualizations=result.get("visualizations", []),
                metrics={
                    "original_shape": result["original_shape"],
                    "final_shape": result["final_shape"]
                },
                config=result.get("config_used", {}),
                logs=["Preprocessing completed successfully"]
            )

            return result

        except Exception as e:
            update_stage_status(
                self.state,
                PipelineStage.PREPROCESSING,
                StageStatus.FAILED,
                error=str(e),
                logs=[f"Error in preprocessing: {str(e)}"]
            )
            raise

    def execute_model_selection(self, problem_type: str) -> Dict[str, Any]:
        """Execute Stage 3: Model Selection"""
        if self.X is None or self.y is None:
            raise ValueError("Preprocessing must be completed before model selection")

        update_stage_status(
            self.state,
            PipelineStage.MODEL_SELECTION,
            StageStatus.IN_PROGRESS,
            logs=["Starting model selection and evaluation..."]
        )

        try:
            result = self.model_selection_agent.select_and_evaluate_models(
                self.X,
                self.y,
                problem_type=problem_type,
                job_id=self.job_id
            )

            # Store best model info
            if result.get("best_model"):
                self.state["selected_models"] = result["models_trained"]
                self.state["best_model"] = result["best_model"]["model_name"]

            update_stage_status(
                self.state,
                PipelineStage.MODEL_SELECTION,
                StageStatus.COMPLETED,
                data=result,
                visualizations=result.get("visualizations", []),
                metrics=result.get("best_model", {}).get("metrics", {}),
                logs=["Model selection completed successfully"]
            )

            return result

        except Exception as e:
            update_stage_status(
                self.state,
                PipelineStage.MODEL_SELECTION,
                StageStatus.FAILED,
                error=str(e),
                logs=[f"Error in model selection: {str(e)}"]
            )
            raise

    def execute_hyperparameter_tuning(self, problem_type: str) -> Dict[str, Any]:
        """Execute Stage 4: Hyperparameter Tuning"""
        if self.model_selection_agent.best_model is None:
            raise ValueError("Model selection must be completed before hyperparameter tuning")

        update_stage_status(
            self.state,
            PipelineStage.HYPERPARAMETER_TUNING,
            StageStatus.IN_PROGRESS,
            logs=["Starting hyperparameter tuning..."]
        )

        try:
            # Split data for training
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )

            # Store split info
            self.state["train_test_split"] = {
                "train_size": len(X_train),
                "test_size": len(X_test),
                "random_state": 42
            }

            result = self.hyperparameter_agent.tune_and_predict(
                model=self.model_selection_agent.best_model,
                model_name=self.model_selection_agent.best_model_name,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                problem_type=problem_type,
                job_id=self.job_id
            )

            # Store tuned model params
            if result.get("status") == "success":
                self.state["model_params"] = result.get("best_params", {})

            update_stage_status(
                self.state,
                PipelineStage.HYPERPARAMETER_TUNING,
                StageStatus.COMPLETED,
                data=result,
                visualizations=result.get("visualizations", []),
                metrics=result.get("metrics", {}),
                config=result.get("best_params", {}),
                logs=["Hyperparameter tuning completed successfully"]
            )

            # Also update prediction stage as completed
            update_stage_status(
                self.state,
                PipelineStage.PREDICTION,
                StageStatus.COMPLETED,
                data={"predictions_available": True},
                logs=["Predictions generated with tuned model"]
            )

            return result

        except Exception as e:
            update_stage_status(
                self.state,
                PipelineStage.HYPERPARAMETER_TUNING,
                StageStatus.FAILED,
                error=str(e),
                logs=[f"Error in hyperparameter tuning: {str(e)}"]
            )
            raise

    def get_state(self) -> PipelineState:
        """Get current pipeline state"""
        return self.state

    def get_stage_result(self, stage: PipelineStage) -> Dict[str, Any]:
        """Get result from a specific stage"""
        return self.state.get(stage.value, {})

    def save_state(self, output_path: str):
        """Save pipeline state to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)

    def get_final_model(self):
        """Get the final tuned model"""
        return self.hyperparameter_agent.get_model()

    def make_predictions(self, X_new: pd.DataFrame) -> pd.Series:
        """Make predictions on new data"""
        model = self.get_final_model()
        if model is None:
            raise ValueError("Pipeline must be executed before making predictions")
        return pd.Series(model.predict(X_new))

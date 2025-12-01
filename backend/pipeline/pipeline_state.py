"""
Pipeline State Management for Data Monkey MVP
Manages the state of the 5-stage data science pipeline
"""
from typing import TypedDict, Optional, Dict, Any, List
from enum import Enum
from datetime import datetime


class PipelineStage(Enum):
    """Enumeration of pipeline stages"""
    DATA_UNDERSTANDING = "data_understanding"
    PREPROCESSING = "preprocessing"
    MODEL_SELECTION = "model_selection"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    PREDICTION = "prediction"


class StageStatus(Enum):
    """Status of each pipeline stage"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageResult(TypedDict, total=False):
    """Result from a pipeline stage"""
    status: str
    timestamp: str
    data: Dict[str, Any]
    visualizations: List[str]
    metrics: Dict[str, Any]
    config: Dict[str, Any]
    error: Optional[str]
    logs: List[str]


class PipelineState(TypedDict, total=False):
    """Complete pipeline state"""
    job_id: str
    created_at: str
    updated_at: str
    current_stage: str

    # Data information
    dataset_path: str
    dataset_info: Dict[str, Any]

    # Stage results
    data_understanding: StageResult
    preprocessing: StageResult
    model_selection: StageResult
    hyperparameter_tuning: StageResult
    prediction: StageResult

    # Processed data at each stage
    original_data_shape: tuple
    preprocessed_data_shape: tuple
    train_test_split: Dict[str, Any]

    # Model artifacts
    selected_models: List[str]
    best_model: Optional[str]
    model_params: Dict[str, Any]

    # Overall metadata
    problem_type: str  # classification or regression
    target_column: str
    feature_columns: List[str]


def create_initial_state(job_id: str, dataset_path: str) -> PipelineState:
    """Create initial pipeline state"""
    now = datetime.now().isoformat()

    return PipelineState(
        job_id=job_id,
        created_at=now,
        updated_at=now,
        current_stage=PipelineStage.DATA_UNDERSTANDING.value,
        dataset_path=dataset_path,
        dataset_info={},
        data_understanding=StageResult(
            status=StageStatus.PENDING.value,
            timestamp=now,
            data={},
            visualizations=[],
            metrics={},
            config={},
            logs=[]
        ),
        preprocessing=StageResult(
            status=StageStatus.PENDING.value,
            timestamp=now,
            data={},
            visualizations=[],
            metrics={},
            config={},
            logs=[]
        ),
        model_selection=StageResult(
            status=StageStatus.PENDING.value,
            timestamp=now,
            data={},
            visualizations=[],
            metrics={},
            config={},
            logs=[]
        ),
        hyperparameter_tuning=StageResult(
            status=StageStatus.PENDING.value,
            timestamp=now,
            data={},
            visualizations=[],
            metrics={},
            config={},
            logs=[]
        ),
        prediction=StageResult(
            status=StageStatus.PENDING.value,
            timestamp=now,
            data={},
            visualizations=[],
            metrics={},
            config={},
            logs=[]
        ),
        selected_models=[],
        best_model=None,
        model_params={},
        problem_type="",
        target_column="",
        feature_columns=[]
    )


def update_stage_status(
    state: PipelineState,
    stage: PipelineStage,
    status: StageStatus,
    data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    visualizations: Optional[List[str]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    logs: Optional[List[str]] = None
) -> PipelineState:
    """Update the status of a pipeline stage"""
    stage_key = stage.value
    now = datetime.now().isoformat()

    state['updated_at'] = now
    state['current_stage'] = stage.value

    stage_result = state.get(stage_key, StageResult())
    stage_result['status'] = status.value
    stage_result['timestamp'] = now

    if data is not None:
        stage_result['data'] = data
    if error is not None:
        stage_result['error'] = error
    if visualizations is not None:
        stage_result['visualizations'] = visualizations
    if metrics is not None:
        stage_result['metrics'] = metrics
    if config is not None:
        stage_result['config'] = config
    if logs is not None:
        stage_result['logs'] = logs

    state[stage_key] = stage_result
    return state

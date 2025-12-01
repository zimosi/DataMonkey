from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging
from pydantic import BaseModel
from typing import Optional, Dict, Any
from agents.summarize import summarize_data

app = FastAPI(title="Data Monkey API - Interactive ML Pipeline")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path("backend/static")
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files for serving plots
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store active pipelines in memory (in production, use a database)
active_pipelines: Dict[str, Any] = {}


class PromptRequest(BaseModel):
    prompt: str
    jobId: str


class PipelineRunRequest(BaseModel):
    jobId: str
    userPrompt: Optional[str] = ""
    targetColumn: Optional[str] = None
    preprocessingConfig: Optional[Dict[str, Any]] = None


class StageConfigRequest(BaseModel):
    jobId: str
    stage: str
    config: Dict[str, Any]


@app.get("/")
async def root():
    return {
        "message": "Data Monkey API - Interactive ML Pipeline",
        "status": "running",
        "version": "1.0.0"
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV file and get initial analysis
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")

        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Save file
        file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read CSV data for quick summary
        df = pd.read_csv(file_path)
        summary = await summarize_data(df, file.filename, job_id)

        logger.info(f"File uploaded: {file.filename} with job_id: {job_id}")

        return JSONResponse(
            status_code=200,
            content={
                "message": "File uploaded successfully",
                "job_id": job_id,
                "filename": file.filename,
                "summary": summary,
                "shape": {
                    "rows": len(df),
                    "columns": len(df.columns)
                },
                "columns": list(df.columns)
            }
        )

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/api/pipeline/run")
async def run_pipeline(request: PipelineRunRequest):
    """
    Execute the complete 5-stage ML pipeline
    """
    try:
        job_id = request.jobId

        logger.info(f"Starting pipeline execution for job {job_id}")

        # Get the uploaded file path
        file_paths = list(UPLOAD_DIR.glob(f"{job_id}_*"))
        if not file_paths:
            raise HTTPException(status_code=404, detail=f"No file found for job {job_id}")

        file_path = str(file_paths[0])

        # Create pipeline orchestrator
        from pipeline.pipeline_orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(job_id, file_path)
        active_pipelines[job_id] = orchestrator

        # Execute pipeline
        state = orchestrator.execute_pipeline(
            user_prompt=request.userPrompt,
            preprocessing_config=request.preprocessingConfig,
            target_column=request.targetColumn,
            auto_mode=True
        )

        logger.info(f"Pipeline execution completed for job {job_id}")

        # Convert state to JSON-serializable format
        response_state = _prepare_state_response(state)

        return JSONResponse(
            status_code=200,
            content={
                "message": "Pipeline executed successfully",
                "jobId": job_id,
                "state": response_state
            }
        )

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")


@app.get("/api/pipeline/state/{job_id}")
async def get_pipeline_state(job_id: str):
    """
    Get the current state of a pipeline
    """
    try:
        if job_id not in active_pipelines:
            raise HTTPException(status_code=404, detail=f"Pipeline not found for job {job_id}")

        orchestrator = active_pipelines[job_id]
        state = orchestrator.get_state()

        response_state = _prepare_state_response(state)

        return JSONResponse(
            status_code=200,
            content={
                "jobId": job_id,
                "state": response_state
            }
        )

    except Exception as e:
        logger.error(f"Failed to get pipeline state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get state: {str(e)}")


@app.post("/api/pipeline/stage/rerun")
async def rerun_stage(request: StageConfigRequest):
    """
    Re-run a specific pipeline stage with new configuration
    """
    try:
        job_id = request.jobId
        stage_name = request.stage
        config = request.config

        if job_id not in active_pipelines:
            raise HTTPException(status_code=404, detail=f"Pipeline not found for job {job_id}")

        orchestrator = active_pipelines[job_id]

        # Re-run the specific stage
        if stage_name == "preprocessing":
            result = orchestrator.execute_preprocessing(
                config=config,
                target_column=orchestrator.state.get("target_column")
            )
        elif stage_name == "model_selection":
            result = orchestrator.execute_model_selection(
                problem_type=orchestrator.state.get("problem_type", "classification")
            )
        elif stage_name == "hyperparameter_tuning":
            result = orchestrator.execute_hyperparameter_tuning(
                problem_type=orchestrator.state.get("problem_type", "classification")
            )
        else:
            raise HTTPException(status_code=400, detail=f"Invalid stage: {stage_name}")

        state = orchestrator.get_state()
        response_state = _prepare_state_response(state)

        return JSONResponse(
            status_code=200,
            content={
                "message": f"Stage {stage_name} re-run successfully",
                "jobId": job_id,
                "state": response_state
            }
        )

    except Exception as e:
        logger.error(f"Failed to rerun stage: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to rerun stage: {str(e)}")


@app.get("/api/pipeline/graph/{job_id}")
async def get_pipeline_graph(job_id: str):
    """
    Get the pipeline graph structure for visualization
    """
    try:
        if job_id not in active_pipelines:
            # Return default graph structure
            return JSONResponse(
                status_code=200,
                content=_get_default_graph_structure()
            )

        orchestrator = active_pipelines[job_id]
        state = orchestrator.get_state()

        # Build graph structure based on current state
        graph = _build_graph_from_state(state)

        return JSONResponse(
            status_code=200,
            content=graph
        )

    except Exception as e:
        logger.error(f"Failed to get pipeline graph: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get graph: {str(e)}")


@app.get("/api/pipeline/dag/{job_id}")
async def get_pipeline_dag(job_id: str):
    """
    Get the dynamic DAG structure with all agents and sub-agents
    """
    try:
        from pipeline.dag_manager import DAGManager

        # Try to load existing DAG
        dag_manager = DAGManager.load_dag(job_id)

        if not dag_manager:
            # Create new DAG if doesn't exist
            dag_manager = DAGManager(job_id)
            dag_manager.save_dag()

        dag_structure = dag_manager.get_dag_structure()

        return JSONResponse(
            status_code=200,
            content=dag_structure
        )

    except Exception as e:
        logger.error(f"Failed to get DAG: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get DAG: {str(e)}")


class ChatRequest(BaseModel):
    jobId: str
    agentId: str
    message: str


@app.post("/api/agent/chat")
async def chat_with_agent(request: ChatRequest):
    """
    Send a chat message to a specific agent
    """
    try:
        from chat.agent_chat_handler import AgentChatHandler

        chat_handler = AgentChatHandler()

        # Get agent state from active pipeline
        agent_state = {}
        if request.jobId in active_pipelines:
            orchestrator = active_pipelines[request.jobId]
            state = orchestrator.get_state()
            agent_state = state.get(request.agentId, {})

        # Load conversation history
        conversation_history = chat_handler.load_conversation(request.jobId, request.agentId)

        # Process chat message
        response = chat_handler.chat_with_agent(
            job_id=request.jobId,
            agent_id=request.agentId,
            user_message=request.message,
            agent_state=agent_state,
            conversation_history=conversation_history
        )

        return JSONResponse(
            status_code=200,
            content=response
        )

    except Exception as e:
        logger.error(f"Chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.get("/api/agent/conversation/{job_id}/{agent_id}")
async def get_conversation(job_id: str, agent_id: str):
    """
    Get conversation history for a specific agent
    """
    try:
        from chat.agent_chat_handler import AgentChatHandler

        chat_handler = AgentChatHandler()
        conversation = chat_handler.load_conversation(job_id, agent_id)

        return JSONResponse(
            status_code=200,
            content={
                "jobId": job_id,
                "agentId": agent_id,
                "conversation": conversation
            }
        )

    except Exception as e:
        logger.error(f"Failed to get conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation: {str(e)}")


@app.post("/api/prompt")
async def prompt(prompt_req: PromptRequest):
    """
    Legacy endpoint for LLM analysis (kept for backward compatibility)
    """
    try:
        user_prompt = prompt_req.prompt
        job_id = prompt_req.jobId

        logger.info(f"Starting data analysis for job {job_id}: {user_prompt}")

        # Get the uploaded file path
        file_paths = list(UPLOAD_DIR.glob(f"{job_id}_*"))
        if not file_paths:
            raise HTTPException(status_code=404, detail=f"No file found for job {job_id}")

        file_path = file_paths[0]
        df = pd.read_csv(file_path)

        # Import and run the workflow
        from graph.DA_workflow import analyze_with_prompt
        workflow_result = await analyze_with_prompt(
            df=df,
            user_prompt=user_prompt,
            job_id=job_id
        )

        return JSONResponse(
            status_code=200,
            content={
                "message": "Analysis completed successfully",
                "jobId": job_id,
                "result": workflow_result.get("response", ""),
                "plots": workflow_result.get("plots", []),
            }
        )

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# Helper functions

def _prepare_state_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare pipeline state for JSON response"""
    import json
    # Convert to JSON and back to handle non-serializable types
    return json.loads(json.dumps(state, default=str))


def _get_default_graph_structure() -> Dict[str, Any]:
    """Get default pipeline graph structure"""
    return {
        "nodes": [
            {
                "id": "data_understanding",
                "label": "Data Understanding",
                "type": "agent",
                "status": "pending",
                "description": "Analyze dataset and understand semantic meaning"
            },
            {
                "id": "preprocessing",
                "label": "Preprocessing",
                "type": "agent",
                "status": "pending",
                "description": "Clean and prepare data"
            },
            {
                "id": "model_selection",
                "label": "Model Selection",
                "type": "agent",
                "status": "pending",
                "description": "Train and evaluate multiple models"
            },
            {
                "id": "hyperparameter_tuning",
                "label": "Hyperparameter Tuning",
                "type": "agent",
                "status": "pending",
                "description": "Optimize model parameters"
            },
            {
                "id": "prediction",
                "label": "Prediction",
                "type": "agent",
                "status": "pending",
                "description": "Generate predictions"
            }
        ],
        "edges": [
            {"from": "data_understanding", "to": "preprocessing"},
            {"from": "preprocessing", "to": "model_selection"},
            {"from": "model_selection", "to": "hyperparameter_tuning"},
            {"from": "hyperparameter_tuning", "to": "prediction"}
        ]
    }


def _build_graph_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build graph structure from pipeline state"""
    stages = [
        "data_understanding",
        "preprocessing",
        "model_selection",
        "hyperparameter_tuning",
        "prediction"
    ]

    stage_labels = {
        "data_understanding": "Data Understanding",
        "preprocessing": "Preprocessing",
        "model_selection": "Model Selection",
        "hyperparameter_tuning": "Hyperparameter Tuning",
        "prediction": "Prediction"
    }

    stage_descriptions = {
        "data_understanding": "Analyze dataset and understand semantic meaning",
        "preprocessing": "Clean and prepare data",
        "model_selection": "Train and evaluate multiple models",
        "hyperparameter_tuning": "Optimize model parameters",
        "prediction": "Generate predictions"
    }

    nodes = []
    for stage in stages:
        stage_data = state.get(stage, {})
        nodes.append({
            "id": stage,
            "label": stage_labels[stage],
            "type": "agent",
            "status": stage_data.get("status", "pending"),
            "description": stage_descriptions[stage],
            "visualizations": stage_data.get("visualizations", []),
            "metrics": stage_data.get("metrics", {}),
            "timestamp": stage_data.get("timestamp", "")
        })

    edges = [
        {"from": "data_understanding", "to": "preprocessing"},
        {"from": "preprocessing", "to": "model_selection"},
        {"from": "model_selection", "to": "hyperparameter_tuning"},
        {"from": "hyperparameter_tuning", "to": "prediction"}
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "currentStage": state.get("current_stage", "data_understanding")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

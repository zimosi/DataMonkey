from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles  # Add for serving static files
import os
import uuid
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging
from agents.summarize import summarize_data

# Import will be done inside the function to avoid circular imports

app = FastAPI(title="Simple CSV Analysis API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Create static directory for plots and mount it
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Artifacts directory for ML models
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

# No need to initialize service - using LangGraph workflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.get("/")
async def root():
    return {"message": "Simple CSV Analysis API", "status": "running"}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV file and get LLM analysis
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
        
        # Read CSV data
        df = pd.read_csv(file_path)
        summary = await summarize_data(df, file.filename, job_id)
        
        # Run LangGraph workflow
        '''
        logger.info(f"Starting LangGraph workflow for {file.filename}")
        from graph.workflow import run_workflow_async
        workflow_result = await run_workflow_async(job_id, str(file_path), file.filename, df)
        
        logger.info(f"LangGraph workflow completed for job {job_id}")
        '''
        return JSONResponse(
            status_code=200,
            content={
                "message": "File uploaded and analyzed successfully.",
                "job_id": job_id,
                "filename": file.filename,
                "llm_analysis": summary,
                "filePath": str(file_path),
            }
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


from pydantic import BaseModel
from typing import Optional

class PromptRequest(BaseModel):
    prompt: str
    jobId: str
    filePath: Optional[str] = None  # Make it optional with default None

@app.post("/api/ml_pipeline")
async def ml_pipeline(prompt_req: PromptRequest):
    """
    Get ML pipeline for a given prompt
    """
    try:
        user_prompt = prompt_req.prompt
        job_id = prompt_req.jobId
        file_path = prompt_req.filePath
        
        logger.info(f"Starting machine learning pipeline for job {job_id}: {user_prompt}")
        
        # Get the uploaded file path based on job_id if not provided
        if not file_path:
            file_paths = list(UPLOAD_DIR.glob(f"{job_id}_*"))
            if not file_paths:
                raise HTTPException(status_code=404, detail=f"No file found for job {job_id}")
            file_path = file_paths[0]
        
        # Read the dataframe
        df = pd.read_csv(file_path)
        
        from graph.ML_workflow import start_machine_learning_pipeline
        ml_result = await start_machine_learning_pipeline(df, job_id, user_prompt)
        
        logger.info(f"ML pipeline completed for job {job_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "ML pipeline completed successfully",
                "jobId": job_id,
                "status": "completed",
                "prompt": user_prompt,
                "result": {
                    "target": ml_result.get("target"),
                    "task": ml_result.get("task"),
                    "schema": ml_result.get("schema"),
                    "reason_profile": ml_result.get("reason_profile"),
                    "preprocess_plan": ml_result.get("preprocess_plan"),
                    "pipeline_path": ml_result.get("pipeline_path"),
                    "splits": ml_result.get("splits"),
                    "hpo_results": ml_result.get("hpo_results"),
                    "best_model_path": ml_result.get("best_model_path"),
                    "candidates": ml_result.get("candidates"),
                }
            }
        )
        
    except FileNotFoundError:
        logger.error(f"File not found for job {job_id}")
        raise HTTPException(status_code=404, detail=f"File not found for job {job_id}")
    except Exception as e:
        logger.error(f"ML pipeline failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ML pipeline failed: {str(e)}")


@app.get("/api/download_model/{job_id}")
async def download_model(job_id: str):
    """
    Download the trained model for a given job_id
    """
    try:
        # Model is saved at artifacts/{job_id}/best_model.pkl
        model_path = ARTIFACTS_DIR / job_id / "best_model.pkl"
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model not found for job {job_id}")
        
        logger.info(f"Downloading model for job {job_id}: {model_path}")
        
        # Return file with appropriate headers for download
        return FileResponse(
            path=str(model_path),
            filename=f"model_{job_id}.pkl",
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="model_{job_id}.pkl"'}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download model for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download model: {str(e)}")
    

@app.post("/api/prompt")
async def prompt(prompt_req: PromptRequest):
    """
    Get LLM analysis for a given prompt
    """
    try:
        user_prompt = prompt_req.prompt
        job_id = prompt_req.jobId
        file_path = prompt_req.filePath
        
        logger.info(f"Starting data analysis for job {job_id}: {user_prompt}")
        # Read the dataframe
        df = pd.read_csv(file_path)
        
        logger.info(f"Running workflow for job {job_id}")
        
        # Import and run the workflow
        from graph.DA_workflow import analyze_with_prompt
        workflow_result = await analyze_with_prompt(
            df=df,
            user_prompt=user_prompt,
            job_id=job_id
        )
        
        logger.info(f"Workflow completed for job {job_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Analysis completed successfully",
                "jobId": job_id,
                "result": workflow_result.get("response", ""),  # Extract the text response
                "plots": workflow_result.get("plots", []),  # Include plot URLs
            }
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")



    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

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
        
        # Run LangGraph workflow
        logger.info(f"Starting LangGraph workflow for {file.filename}")
        from graph.workflow import run_workflow_async
        workflow_result = await run_workflow_async(job_id, str(file_path), file.filename, df)
        
        logger.info(f"LangGraph workflow completed for job {job_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "File uploaded and analyzed successfully.",
                "job_id": job_id,
                "filename": file.filename,
                "status": workflow_result.get("status", "completed"),
                "llm_analysis": workflow_result.get("llm_analysis"),
                "data_summary": workflow_result.get("data_summary"),
                "error": workflow_result.get("error")
            }
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

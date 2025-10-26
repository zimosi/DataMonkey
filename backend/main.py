from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
            }
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


from pydantic import BaseModel

class PromptRequest(BaseModel):
    prompt: str
    jobId: str

@app.post("/api/prompt")
async def prompt(prompt_req: PromptRequest):
    """
    Get LLM analysis for a given prompt
    """
    try:
        user_prompt = prompt_req.prompt
        job_id = prompt_req.jobId
        
        logger.info(f"Starting data analysis for job {job_id}: {user_prompt}")
        
        # Get the uploaded file path based on job_id
        # Since we saved files with pattern: {job_id}_{filename}
        file_paths = list(UPLOAD_DIR.glob(f"{job_id}_*"))
        if not file_paths:
            raise HTTPException(status_code=404, detail=f"No file found for job {job_id}")
        
        file_path = file_paths[0]
        
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

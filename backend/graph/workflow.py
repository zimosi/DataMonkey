import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, TypedDict, Optional
import pandas as pd
from langgraph.graph import StateGraph, END
from agents.summarize_agent import summarize_agent

logger = logging.getLogger(__name__)

# Define the state structure
class WorkflowState(TypedDict):
    job_id: str
    file_path: str
    filename: str
    raw_data: pd.DataFrame
    llm_analysis: Optional[Dict[str, Any]]
    data_summary: Optional[Dict[str, Any]]
    error: Optional[str]
    status: str
    started_at: datetime
    completed_at: Optional[datetime]

def create_workflow() -> StateGraph:
    """Create the LangGraph workflow"""
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("summarize", summarize_agent)
    
    # Set entry point
    workflow.set_entry_point("summarize")
    
    # Add edges
    workflow.add_edge("summarize", END)
    
    return workflow.compile()

async def run_workflow_async(job_id: str, file_path: str, filename: str, df) -> Dict[str, Any]:
    """Run the workflow asynchronously with the given data"""
    try:
        logger.info(f"Starting workflow for job {job_id}")
        
        # Create initial state
        initial_state = {
            "job_id": job_id,
            "file_path": file_path,
            "filename": filename,
            "raw_data": df,
            "llm_analysis": None,
            "data_summary": None,
            "error": None,
            "status": "started",
            "started_at": datetime.now(),
            "completed_at": None
        }
        
        # Create and run the workflow
        workflow = create_workflow()
        final_state = await workflow.ainvoke(initial_state)
        
        # Add completion timestamp
        final_state["completed_at"] = datetime.now()
        
        logger.info(f"Workflow completed for job {job_id}")
        return final_state
        
    except Exception as e:
        logger.error(f"Workflow failed for job {job_id}: {str(e)}")
        return {
            "job_id": job_id,
            "file_path": file_path,
            "filename": filename,
            "raw_data": df,
            "llm_analysis": None,
            "data_summary": None,
            "error": str(e),
            "status": "failed",
            "started_at": datetime.now(),
            "completed_at": datetime.now()
        }

def run_workflow(job_id: str, file_path: str, filename: str, df) -> Dict[str, Any]:
    """Run the workflow with the given data (sync wrapper)"""
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in a running event loop, create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, run_workflow_async(job_id, file_path, filename, df))
                return future.result()
        else:
            # If no event loop is running, we can use asyncio.run
            return asyncio.run(run_workflow_async(job_id, file_path, filename, df))
    except Exception as e:
        logger.error(f"Workflow execution failed for job {job_id}: {str(e)}")
        return {
            "job_id": job_id,
            "file_path": file_path,
            "filename": filename,
            "raw_data": df,
            "llm_analysis": None,
            "data_summary": None,
            "error": str(e),
            "status": "failed",
            "started_at": datetime.now(),
            "completed_at": datetime.now()
        }

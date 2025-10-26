import asyncio
import logging
from datetime import datetime
from typing import TypedDict, List, Annotated, Dict, Any
import operator
import pandas as pd
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from typing import Literal, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI issues
import matplotlib.pyplot as plt
import uuid
import os
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_MAX_TOKENS, OPENAI_TEMPERATURE

logger = logging.getLogger(__name__)

DATA: dict[str, pd.DataFrame] = {}

def seed_data():
    """Initialize artifacts directory"""
    os.makedirs("static", exist_ok=True)
    os.makedirs("static/plots", exist_ok=True)  # For serving plots to frontend

# -----------------------------
# Tools (LLM decides to call)
# -----------------------------
'''
@tool("describe_table")
def describe_table(df_ref: str) -> str:
    """
    Return a compact summary of the dataset: columns + basic stats.
    Use when the user asks for an overview or schema.
    """
    if df_ref not in DATA:
        return f"Dataset '{df_ref}' not found."
    df = DATA[df_ref]
    cols = ", ".join(f"{c}({df[c].dtype})" for c in df.columns)
    stats = df.describe(include="all").to_string()
    return f"Columns: {cols}\n\nStats:\n{stats}"
'''
@tool("plot")
def plot(
    job_id: str,
    kind: Literal["hist","scatter","line"],
    x: str,
    y: Optional[str] = None,
    bins: int = 30,
    title: Optional[str] = None
) -> str:
    """
    Generate a matplotlib plot and return the saved image path.
    Call this ONLY if a visual clearly helps answer the user question or user explicitly asks for a plot.
    """
    try:
        if job_id not in DATA:
            return f"Dataset '{job_id}' not found."
        df = DATA[job_id]
        if x not in df.columns or (y and y not in df.columns):
            return f"Invalid columns. Available: {list(df.columns)}"

        fig = plt.figure(figsize=(10, 6))
        if kind == "hist":
            df[x].plot(kind="hist", bins=bins)
            plt.xlabel(x)
        elif kind == "scatter":
            if y is None:
                return "Scatter requires 'y'."
            plt.scatter(df[x], df[y])
            plt.xlabel(x); plt.ylabel(y)
        else:  # line
            if y is None:
                return "Line requires 'y'."
            plt.plot(df[x], df[y])
            plt.xlabel(x); plt.ylabel(y)

        if title:
            plt.title(title)

        out = f"static/plots/plot_{uuid.uuid4().hex[:8]}.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close(fig)
        return f"Plot saved to: {out}"
    except Exception as e:
        logger.error(f"Plot generation failed: {e}")
        return f"Error creating plot: {str(e)}"

TOOLS = [plot]

# -----------------------------
# Graph state
# -----------------------------
class State(TypedDict):
    messages: Annotated[list, operator.add]
    df_ref: str  # current dataset handle
    plots: Annotated[list[str], operator.add]
    data: pd.DataFrame
    job_id: str
    user_prompt: str
    step_count: int  # Track iteration count

# -----------------------------
# LLM with tools (implicit call)
# -----------------------------
SYSTEM = SystemMessage(
    content=(
        "You are a helpful data analyst assistant. "
        "Analyze the data and answer user questions clearly and concisely.\n\n"
        "Tools available:\n"
        "- `plot`: Create visualizations (ONLY use if user explicitly asks for a plot/chart)\n\n"
        "CRITICAL RULES:\n"
        "- Answer most questions directly WITHOUT calling tools\n"
        "- Only call `plot` if the user explicitly asks for a visualization\n"
        "- After the tool returns a result, you MUST immediately provide a final answer to the user\n"
        "- Do NOT call the same tool again after seeing its result\n"
        "- Your final answer should reference the plot URL if one was created\n"
        "- Be concise and helpful\n"
        "- STOP immediately after giving your final answer"
    )
)

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=OPENAI_TEMPERATURE,
    max_tokens=OPENAI_MAX_TOKENS
).bind_tools(TOOLS)

def llm_node(state: State) -> State:
    """LLM node that can call tools"""
    
    # Increment and log step count
    step = state.get("step_count", 0) + 1
    logger.info(f"\n{'='*50}")
    logger.info(f"📊 STEP {step}: LLM Node")
    logger.info(f"{'='*50}")
    
    # Get the last user message
    user_message = state["messages"][-1].content if state["messages"] else state["user_prompt"]
    
    df = state['data']
    csv_preview = df.head(20).to_string()
    
    # Check if there's a tool result in the messages
    last_msg = state["messages"][-1] if state["messages"] else None
    has_tool_result = last_msg and isinstance(last_msg, ToolMessage)
    
    # Create enhanced prompt with context
    if has_tool_result:
        # After tool execution, force the LLM to provide a final answer
        enhanced_prompt = f"""The tool has been executed successfully. Now provide a final answer to the user's question.

Tool result: {last_msg.content}

User's original question: {state['user_prompt']}

Provide a brief, helpful answer that references the tool result."""
    else:
        # Initial prompt before any tools
        enhanced_prompt = f"""You are analyzing a dataset with {len(df)} rows and {len(df.columns)} columns.

Columns: {', '.join(df.columns.tolist())}

job_id: {state['job_id']}

Data preview:
{csv_preview}

User question: {user_message}

Provide a helpful analysis or use tools if needed for visualization."""
    
    logger.info(f"💬 User question: {user_message}")
    
    # Invoke LLM with enhanced prompt
    ai = llm.invoke([SYSTEM, HumanMessage(content=enhanced_prompt)])
    
    # Log the AI's response
    logger.info(f"🤖 AI thinking...")
    logger.info(f"💭 AI Response: {ai.content[:200]}...")
    if hasattr(ai, 'tool_calls') and ai.tool_calls:
        logger.info(f"🔧 AI wants to call tools: {[tc.get('name') for tc in ai.tool_calls]}")
    
    return {"messages": [ai], "step_count": step}

# Create ToolNode once at module level
tool_executor = ToolNode(TOOLS)

# Custom tool node wrapper to add logging
def tool_node_wrapper(state: State) -> State:
    """Execute tools and log the results"""
    logger.info("⚙️ Executing tools...")
    
    # Execute tools using invoke method
    result = tool_executor.invoke(state)
    
    # Log tool results
    last_msg = result["messages"][-1]
    logger.info(f"🔧 Tool result: {last_msg.content[:200] if hasattr(last_msg, 'content') else 'No content'}...")
    
    # Extract plot URL from tool result and add to state
    if hasattr(last_msg, 'content') and "Plot saved to:" in last_msg.content:
        # Extract the plot path from the tool result
        plot_path = last_msg.content.split("Plot saved to: ")[1].strip()
        # Append to existing plots list
        current_plots = result.get("plots", [])
        current_plots.append(plot_path)
        result["plots"] = current_plots
        logger.info(f"📊 Saved plot URL to state: {plot_path}")
    
    return result

tool_node = tool_node_wrapper

def route_after_llm(state: State) -> str:
    """Route based on whether the LLM wants to call tools"""
    step_count = state.get("step_count", 0)
    
    # Force stop after too many iterations to prevent infinite loops
    if step_count >= 5:
        logger.info(f"🛑 Max steps ({step_count}) reached, forcing end")
        return END
    
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        logger.info("🔄 Routing to tools...")
        return "tools"
    logger.info("✅ No tools needed, ending workflow")
    return END

def create_workflow() -> StateGraph:
    """Create the LangGraph workflow"""
    graph = StateGraph(State)
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_node)

    # Set entry point
    graph.set_entry_point("llm")
    
    # Add conditional edges
    graph.add_conditional_edges("llm", route_after_llm, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")  # after tools run, return to LLM to finish

    return graph.compile()

async def analyze_with_prompt(df: pd.DataFrame, user_prompt: str, job_id: str) -> str:
    """
    Main entry point for data analysis with a user prompt.
    
    Args:
        df: The dataframe to analyze
        user_prompt: The user's question/prompt
        job_id: Unique identifier for this analysis
    
    Returns:
        str: The final analysis result
    """
    try:
        logger.info(f"Starting analysis for job {job_id} with prompt: {user_prompt}")
        
        # Initialize artifacts directory
        seed_data()
        
        # Store dataframe in DATA with job_id as key
        DATA[job_id] = df
        
        # Create workflow with recursion limit
        workflow = create_workflow()
        
        # Initial state with user message
        initial_state = {
            "messages": [HumanMessage(content=user_prompt)],
            "df_ref": job_id,
            "plots": [],
            "data": df,
            "job_id": job_id,
            "user_prompt": user_prompt,
            "step_count": 0
        }
        
        # Run workflow with config
        logger.info("Running workflow...")
        config = {"recursion_limit": 10}  # Limit iterations
        final_state = await workflow.ainvoke(initial_state, config=config)
        
        # Extract final response from last message
        last_message = final_state["messages"][-1]
        response_content = last_message.content
        plots = final_state.get("plots", [])  # Get the list of plot URLs
        
        logger.info(f"Analysis completed for job {job_id}")
        
        # Clean up: remove dataframe from DATA
        if job_id in DATA:
            del DATA[job_id]
        
        return {"response": response_content, "plots": plots}
        
    except Exception as e:
        logger.error(f"Analysis failed for job {job_id}: {str(e)}")
        # Clean up on error
        if job_id in DATA:
            del DATA[job_id]
        raise

# Keep the old functions for backward compatibility
async def run_workflow_async(job_id: str, file_path: str, filename: str, df) -> Dict[str, Any]:
    """Legacy function - kept for backward compatibility"""
    logger.warning("Using legacy run_workflow_async - consider using analyze_with_prompt instead")
    try:
        logger.info(f"Starting workflow for job {job_id}")
        
        # Store data
        seed_data()
        DATA[job_id] = df
        
        # Create workflow
        workflow = create_workflow()
        
        # Initial state
        initial_state = {
            "messages": [HumanMessage(content=f"Analyze the dataset from {filename}")],
            "df_ref": job_id,
            "plots": []
        }
        
        # Run workflow
        final_state = await workflow.ainvoke(initial_state)
        
        # Extract response
        last_message = final_state["messages"][-1]
        
        # Clean up
        if job_id in DATA:
            del DATA[job_id]
        
        return {
            "job_id": job_id,
            "response": last_message.content,
            "status": "completed",
            "completed_at": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Workflow failed for job {job_id}: {str(e)}")
        if job_id in DATA:
            del DATA[job_id]
        raise

def run_workflow(job_id: str, file_path: str, filename: str, df) -> Dict[str, Any]:
    """Run the workflow with the given data (sync wrapper)"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, run_workflow_async(job_id, file_path, filename, df))
                return future.result()
        else:
            return asyncio.run(run_workflow_async(job_id, file_path, filename, df))
    except Exception as e:
        logger.error(f"Workflow execution failed for job {job_id}: {str(e)}")
        raise

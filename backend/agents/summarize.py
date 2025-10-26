import pandas as pd
import logging
from typing import Dict, Any
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_MAX_TOKENS, OPENAI_TEMPERATURE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def _call_openai_api(prompt: str) -> str:
    """Call OpenAI API with the analysis prompt"""
    api_key = OPENAI_API_KEY
    model = OPENAI_MODEL
    max_tokens = OPENAI_MAX_TOKENS
    temperature = OPENAI_TEMPERATURE
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert data analyst. Provide clear, insightful analysis of datasets."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Return the text response directly
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"OpenAI API call failed: {str(e)}")
        raise

async def summarize_data(df: pd.DataFrame, filename: str, job_id: str) -> str:
    """Process the data and generate summary using LLM"""


    if df is None:
        raise ValueError("No data provided for summarization")
    
    # Convert dataframe to string representation
    csv_preview = df.head(20).to_string()  # Show first 20 rows
    
    # Create simple prompt
    prompt = f"""Please analyze this CSV dataset and provide a short summary in 100 words.

Dataset: {filename}
Number of rows: {len(df)}
Number of columns: {len(df.columns)}

Column names: {', '.join(df.columns.tolist())}

Data preview:
{csv_preview}
"""
    
    # Call OpenAI API
    
    llm_response = await _call_openai_api(prompt)
    
    return llm_response
    


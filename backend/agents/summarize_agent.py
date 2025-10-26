import pandas as pd
import logging
from typing import Dict, Any
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_MAX_TOKENS, OPENAI_TEMPERATURE

logger = logging.getLogger(__name__)

class SummarizeAgent:
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.model = OPENAI_MODEL
        self.max_tokens = OPENAI_MAX_TOKENS
        self.temperature = OPENAI_TEMPERATURE
        self.client = OpenAI(api_key=self.api_key)
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the data and generate summary using LLM"""
        try:
            logger.info(f"Starting summarization for job {state['job_id']}")
            
            # Get data from state
            df = state.get('raw_data')
            filename = state.get('filename', 'unknown')
            
            if df is None:
                raise ValueError("No data provided for summarization")
            
            # Convert dataframe to string representation
            csv_preview = df.head(20).to_string()  # Show first 20 rows
            
            # Create simple prompt
            prompt = f"""Please analyze this CSV dataset and provide a comprehensive summary.

Dataset: {filename}
Number of rows: {len(df)}
Number of columns: {len(df.columns)}

Column names: {', '.join(df.columns.tolist())}

Data preview:
{csv_preview}
"""
            
            # Call OpenAI API
            llm_response = await self._call_openai_api(prompt)
            
            # Update state with results
            state['llm_analysis'] = llm_response
            state['status'] = 'summarized'
            state['data_summary'] = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist()
            }
            
            logger.info(f"Summarization completed for job {state['job_id']}")
            return state
            
        except Exception as e:
            logger.error(f"Summarization failed for job {state['job_id']}: {str(e)}")
            state['error'] = str(e)
            state['status'] = 'failed'
            return state
    
    async def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API with the analysis prompt"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert data analyst. Provide clear, insightful analysis of datasets."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Return the text response directly
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise

# LangGraph node function
async def summarize_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node function for the summarize agent"""
    agent = SummarizeAgent()
    return await agent.process(state)

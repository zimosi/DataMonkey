from summarize import _call_openai_api

def general_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = state.get('prompt')
    prompt = f"""Please analyze this CSV dataset and reply to the user's prompt: {user_prompt}

Dataset: {filename}
Number of rows: {len(df)}
Number of columns: {len(df.columns)}

Column names: {', '.join(df.columns.tolist())}

Data preview:
{csv_preview}

Give the output in a json format with the following keys:
answer: the answer to the user's prompt
Plot: True or False, if the user's prompt is related to the data, return True, otherwise return False

"""
    
    llm_response = await _call_openai_api(prompt)
    return llm_response
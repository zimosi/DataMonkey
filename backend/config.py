import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
# Validate required configuration
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required. Please set it in your .env file or environment.")


def get_openai_config():
    """
    Get OpenAI configuration as a dictionary

    Returns:
        dict: Configuration dictionary with model, temperature, and max_tokens
    """
    return {
        "model": OPENAI_MODEL,
        "temperature": OPENAI_TEMPERATURE,
        "max_tokens": OPENAI_MAX_TOKENS,
        "api_key": OPENAI_API_KEY
    }

import os
from dotenv import load_dotenv

load_dotenv()

# Database Configurations
POSTGRES_URI = os.getenv("POSTGRES_URI", "postgresql://postgres:password@localhost:5433/ecommerce")


# LLM API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

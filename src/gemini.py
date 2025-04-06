import os
from dotenv import load_dotenv
import google.generativeai as genai

from pydantic import BaseModel
from typing import Literal, List

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

DEFAULT_MODEL = "gemini-2.0-flash"


class History(BaseModel):
    role: Literal["system", "user", "model"]
    parts: List[str]


def initialize_gemini_model(model_name: str = DEFAULT_MODEL):
    return genai.GenerativeModel(model_name=model_name)


_model_cache = {}


def get_gemini_model(model_name: str = DEFAULT_MODEL):
    if model_name not in _model_cache:
        _model_cache[model_name] = initialize_gemini_model(model_name)
    return _model_cache[model_name]


def generate(
    prompt: List[History],
    model_name: str = DEFAULT_MODEL,
) -> str:
    print(prompt)
    try:
        model = get_gemini_model(model_name)

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        error_message = f"Gemini APIエラー: {str(e)}"
        return error_message

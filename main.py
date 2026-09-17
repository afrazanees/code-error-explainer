import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = FastAPI(title="Code Error Explainer")


def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured. Add it to .env or system environment variables.",
        )
    return genai.Client(api_key=api_key)


class ExplainRequest(BaseModel):
    input: str


@app.post("/explain")
async def explain_error(request: ExplainRequest):
    user_input = request.input.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Input cannot be empty.")

    client = get_client()

    # Under-100-word constraint ensures concise root-cause diagnosis without conversational padding
    system_prompt = (
        "You are a calm, precise debugging assistant. "
        "Given an error message or code snippet, explain in plain English what's likely going wrong, "
        "then suggest the most probable fix. Keep it under 100 words unless code is required."
    )

    response = None
    last_err = None
    for model_name in ["gemini-2.5-flash", "gemini-2.5-flash-lite"]:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_input,
                config={"system_instruction": system_prompt},
            )
            break
        except Exception as err:
            last_err = err

    if response is None:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {last_err}")

    return {"explanation": response.text}


app.mount("/", StaticFiles(directory="static", html=True), name="static")

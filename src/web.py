from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from content_generation.edu_content_perplexity import generate_text, generate_questions

class GenerateRequest(BaseModel):
    topic: str

class GenerateResponse(BaseModel):
    text: str
    questions: List[str]


class GenerateHelpResponse(BaseModel):
    help: str


app = FastAPI()

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required")

    # Generate text and questions using existing functions
    text = generate_text(topic)
    questions = generate_questions(text)

    return GenerateResponse(text=text, questions=questions)


# Use this endpoint to get API documentation.
# Note that `docs` is a reserved endpoind in FastAPI used for Swagger UI.
@app.get("/help", response_model=GenerateHelpResponse)
async def help():
    return GenerateHelpResponse(help="This is a Japanese Language Exercise System (JES)")

import json
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from pathlib import Path
from uuid import uuid4
from pydantic import BaseModel

from content_generation.edu_content_local import generate_text, generate_questions
from content_generation.vocabulary import default_text, default_questions
from assessment.analysis import analyse_answers

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
    print(f"Generated text for topic '{topic}': {text}")
    questions = generate_questions(text)
    print(f"Generated questions for topic '{topic}': {questions}")

    return GenerateResponse(text=text, questions=questions)


# Use this endpoint to get API documentation.
# Note that `docs` is a reserved endpoind in FastAPI used for Swagger UI.
@app.get("/help", response_model=GenerateHelpResponse)
async def help():
    return GenerateHelpResponse(help="This is a Japanese Language Exercise System (JES)")


# Returns a canned response for testing the text generation workflow.
@app.post("/generate-test", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required")

    # Use a canned response for testing
    test_response = json.loads("""
    {"text": "今日は母の誕生日です。誕生日、おめでとうございます。お母さんは四十五歳ぐらいの歳です。", "questions": ["今日は**お母さん**の**誕生日**ですか。", "お母さん は**何歳**ぐらいですか。", "**姉**は何が**とくい**ですか。"]}
    """)
    text = test_response["text"]
    questions = test_response["questions"]

    return GenerateResponse(text=text, questions=questions)



UPLOAD_DIR = Path(__file__).parent / "uploads" / "answers"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/submit-answer")
async def submit_answer(
    text: str = Form(...),
    selected_question: str = Form(...),
    handwritten: UploadFile = File(...),
    spoken: UploadFile = File(...),
):
    # Basic validation of required form fields
    if not text.strip():
        raise HTTPException(status_code=400, detail="`text` is required")
    if not selected_question.strip():
        raise HTTPException(status_code=400, detail="`selected_question` is required")

    # Validate content types/extensions
    hw_suffix = Path(handwritten.filename).suffix.lower()
    sp_suffix = Path(spoken.filename).suffix.lower()
    if hw_suffix != ".png" and not (handwritten.content_type and "image" in handwritten.content_type):
        raise HTTPException(status_code=400, detail="Handwritten file must be a PNG image")
    if sp_suffix not in (".wav",) and not (spoken.content_type and "wav" in spoken.content_type):
        raise HTTPException(status_code=400, detail="Spoken file must be a WAV audio")

    # Save files with unique names
    hw_path = UPLOAD_DIR / f"{uuid4().hex}{hw_suffix or '.png'}"
    sp_path = UPLOAD_DIR / f"{uuid4().hex}{sp_suffix or '.wav'}"

    try:
        hw_bytes = await handwritten.read()
        sp_bytes = await spoken.read()

        hw_path.write_bytes(hw_bytes)
        sp_path.write_bytes(sp_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded files: {e}")

    # Call analyse_answers in threadpool (analysis is synchronous / may block)
    try:
        feedback = await run_in_threadpool(
            analyse_answers, text, selected_question, str(hw_path), str(sp_path)
        )
        # feedback = "All good! ありがとうございます。"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    return Response(content=feedback, media_type="text/plain; charset=utf-8")

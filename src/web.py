"""
This module provides a FastAPI-based REST API for the Japanese Language Exercise System.
The API serves as the backend that can be accessed via HTTP requests from a frontend web or mobile application.
"""

import json
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from pathlib import Path
from uuid import uuid4
from pydantic import BaseModel

from content_generation.edu_content_local import generate_text, generate_questions
from assessment.analysis_local import analyse_answers


# =============================================================================
# Pydantic models -- request/response schemas
# =============================================================================

class GenerateRequest(BaseModel):
    """Request model for content generation endpoints.
    
    Attributes:
        topic: The subject/theme for which the Japanese text and questions
               should be generated (e.g., "family", "weather", "school").
    """
    topic: str


class GenerateResponse(BaseModel):
    """Response model for content generation endpoints.
    
    Attributes:
        text: A Japanese text generated based on the requested topic.
              This text serves as reading material for the learner.
        questions: A list of comprehension questions about the generated text.
    """
    text: str
    questions: List[str]


class GenerateHelpResponse(BaseModel):
    """Response model for the help endpoint.
    
    Attributes:
        help: A brief description of the API and its purpose.
    """
    help: str


# =============================================================================
# FastAPI application instance
# =============================================================================

app = FastAPI()


# =============================================================================
# API endpoints
# =============================================================================

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate Japanese text and questions based on a given topic.
    
    This is the main content generation endpoint with the following workflow:
    
    1. Extract and validate the topic from the request body
    2. Create a Japanese passage about the topic using a local LLM
    3. Generate comprehension questions based on the content of the text
    4. Log the generated content for debugging purposes
    5. Return both text and questions in a structured response
    
    Args:
        req: GenerateRequest containing the topic string
        
    Returns:
        GenerateResponse with generated Japanese text and list of questions
        
    Raises:
        HTTPException(400): If topic is empty or contains only whitespace
    """
    # Step 1: Validate input -- ensure topic is not empty after stripping whitespace
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required")

    # Step 2: Generate Japanese educational text using the local LLM.
    text = generate_text(topic)
    # Show the generated text in the console for debugging and verification purposes
    print(f"Generated text for topic '{topic}': {text}")
    
    # Step 3: Generate comprehension questions based on the generated text
    questions = generate_questions(text)
    # Show the generated questions in the console for debugging and verification purposes
    print(f"Generated questions for topic '{topic}': {questions}")

    # Step 4: Return structured response with text and questions
    return GenerateResponse(text=text, questions=questions)


@app.get("/help", response_model=GenerateHelpResponse)
async def help():
    """Provide basic API documentation and system description.
    
    This is a simple informational endpoint maily for verifying that the server is running and
    is available to handle requests.
    
    Note:
        FastAPI automatically provides interactive API documentation at /docs
        (Swagger UI) and /redoc (ReDoc). This endpoint provides a simpler,
        custom help message.
    
    Returns:
        GenerateHelpResponse with a brief system description
    """
    return GenerateHelpResponse(
        help=(
            "This is a Japanese Language Exercise System (JES). "
            "Use the /generate endpoint to create Japanese text and questions based on a topic, "
            "and /submit-answer to submit answers for analysis."
        )
    )


@app.post("/generate-test", response_model=GenerateResponse)
async def generate_test(req: GenerateRequest):
    """Test endpoint that returns canned/mock content for development and testing.
    
    Provides a predictable, consistent response for testing the frontend
    or integration workflows without invoking the actual LLM-based content
    generation. This is useful for:
    - Frontend development without backend dependencies
    - Integration testing with known expected values
    - Debugging the API workflow without LLM latency

    Args:
        req: GenerateRequest containing the topic string (validated but not used)
        
    Returns:
        GenerateResponse with predefined test content:
        - text: A Japanese passage about a mother's birthday
        - questions: Three comprehension questions
        
    Raises:
        HTTPException(400): If topic is empty or contains only whitespace
    """
    # Step 1: Validate input for API consistency with the real /generate endpoint
    # (even though we ignore the topic here)
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required")

    # Step 2: Parse canned response - hardcoded test data
    test_response = json.loads("""
    {"text": "今日は母の誕生日です。誕生日、おめでとうございます。お母さんは四十五歳ぐらいの歳です。", "questions": ["今日は**お母さん**の**誕生日**ですか。", "お母さん は**何歳**ぐらいですか。", "**姉**は何が**とくい**ですか。"]}
    """)
    
    # Step 3: Extract text and questions from the parsed JSON
    text = test_response["text"]
    questions = test_response["questions"]

    # Step 4: Return the canned response in the standard format
    return GenerateResponse(text=text, questions=questions)


# =============================================================================
# Answer submission and analysis endpoint
# =============================================================================

# Define the directory where uploaded answer files (images, audio) will be stored on the server.
# Uses the 'uploads/answers' subdirectory relative to this module's location
UPLOAD_DIR = Path(__file__).parent / "uploads" / "answers"
# Ensure the upload directory exists, creating parent directories if needed
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/submit-answer")
async def submit_answer(
    text: str = Form(...),
    selected_question: str = Form(...),
    handwritten: UploadFile = File(...),
    spoken: UploadFile = File(...),
):
    """Submit and analyze user answers including handwritten and spoken responses.
    
    Args:
        text: The original Japanese text that was shown to the user
              (needed for context during answer analysis)
        selected_question: The specific question the user is answering
                           (used to determine expected answer)
        handwritten: PNG image file containing the user's handwritten
                     Japanese answer (will be processed via OCR)
        spoken: WAV audio file containing the user's spoken Japanese
                answer (will be processed via ASR).
    
    Returns:
        Response: Plain text feedback on the user's answers.

    Raises:
        HTTPException(400): If required fields are empty or file types are invalid
        HTTPException(500): If file saving fails or analysis encounters an error
    
    File Storage:
        Uploaded files are saved with UUID-based filenames to:
        - Prevent filename collisions from concurrent uploads
        - Avoid issues with special characters in original filenames
        - Enable potential future reference or debugging
    """
    # -------------------------------------------------------------------------
    # Step 1: Validate required form fields: text and selected question
    # -------------------------------------------------------------------------
    if not text.strip():
        raise HTTPException(status_code=400, detail="`text` is required")
    if not selected_question.strip():
        raise HTTPException(status_code=400, detail="`selected_question` is required")


    # -------------------------------------------------------------------------
    # Step 2: Validate uploaded file types
    # -------------------------------------------------------------------------
    # Extract file extensions for type checking
    hw_suffix = Path(handwritten.filename).suffix.lower()
    sp_suffix = Path(spoken.filename).suffix.lower()
    
    # Validate handwritten answer is a PNG image by
    # checking both extension and MIME type for robust validation
    if hw_suffix != ".png" and not (handwritten.content_type and "image" in handwritten.content_type):
        raise HTTPException(status_code=400, detail="Handwritten file must be a PNG image")
    
    # Validate spoken answer is a WAV audio file for consistent ASR processing
    if sp_suffix not in (".wav",) and not (spoken.content_type and "wav" in spoken.content_type):
        raise HTTPException(status_code=400, detail="Spoken file must be a WAV audio")

    # -------------------------------------------------------------------------
    # Step 3: Save uploaded files with unique identifiers
    # -------------------------------------------------------------------------
    # Generate unique file paths using UUID to prevent naming collisions
    hw_path = UPLOAD_DIR / f"{uuid4().hex}{hw_suffix or '.png'}"
    sp_path = UPLOAD_DIR / f"{uuid4().hex}{sp_suffix or '.wav'}"

    try:
        # Asynchronously read file contents from the upload stream
        hw_bytes = await handwritten.read()
        sp_bytes = await spoken.read()

        # Write files to disk for processing by the analysis module
        hw_path.write_bytes(hw_bytes)
        sp_path.write_bytes(sp_bytes)
    except Exception as e:
        # Handle any file I/O errors during the save process
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded files: {e}")

    # -------------------------------------------------------------------------
    # Step 4: Analyze the submitted answers
    # -------------------------------------------------------------------------
    try:
        # Run analysis in a thread pool because analyse_answers() is synchronous
        # and may involve blocking operations (OCR, ASR, LLM calls).
        # This prevents blocking the async event loop.
        feedback = await run_in_threadpool(
            analyse_answers, text, selected_question, str(hw_path), str(sp_path)
        )
    except Exception as e:
        # Handle any errors during the analysis process
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    # -------------------------------------------------------------------------
    # Step 5: Return feedback to the user
    # -------------------------------------------------------------------------
    # Return plain text response with explicit UTF-8 encoding
    # critical for proper display of Japanese characters
    return Response(content=feedback, media_type="text/plain; charset=utf-8")

"""
Timed analysis benchmarking (Perplexity API)
============================================

Purpose:
    Benchmarks the performance of the Perplexity API for analysing user's answers.
    Measures execution time for the cloud-based analysis pipeline, providing a
    comparison baseline against local model performance.

Workflow:
    1. Loads API credentials from environment variables (.env file)
    2. Runs the answer analysis pipeline using Perplexity API which:
       - Processes a handwritten answer image (locally via OCR)
       - Processes a spoken answer audio file (locally via ASR)
       - Sends OCR/ASR results to Perplexity for evaluation
    3. Logs detailed results and execution time to a log file

Output:
    Creates a single log file:
        logs/perplexity-assessment.log

Requirements:
    - PERPLEXITY_API_KEY environment variable must be set
    - Internet connection for API access

Usage:
    python -m src.scripts.timed_analysis_perplexity
"""

import time

from pathlib import Path

from dotenv import load_dotenv

from content_generation.vocabulary import default_text, default_questions
from assessment.analysis_perplexity import analyse_answers
from scripts.utils import format_duration

# Load environment variables (PERPLEXITY_API_KEY) from .env file
load_dotenv()

CURRENT_MODULE_DIRPATH = Path(__file__).parent.resolve()
LOG_DIRPATH = CURRENT_MODULE_DIRPATH.parent.parent / "logs"


log_filepath = LOG_DIRPATH / "perplexity-assessment.log"
start = time.perf_counter()

analyse_answers(
    text=default_text,
    question=default_questions[0],
    handwritten_answer_filepath=CURRENT_MODULE_DIRPATH.parent / "assessment/model-answers/answer-0.png",
    spoken_answer_filepath=CURRENT_MODULE_DIRPATH.parent / "assessment/model-answers/answer-0.wav",
    log_filepath=log_filepath,
)

elapsed = time.perf_counter() - start
with open(log_filepath, "a", encoding="utf-8") as f:
    f.write(f"Execution time: {format_duration(elapsed)}\n\n")

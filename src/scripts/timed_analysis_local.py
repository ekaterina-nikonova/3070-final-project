"""
Timed analysis benchmarking (local models)
==========================================

Purpose:
    Benchmarks the performance of local LLM models for analysing user's answers.
    Measures execution time for each model to evaluate answer correctness against
    a reference text and question. Used for comparing inference speeds across
    different model sizes.

Workflow:
    1. Detects CUDA availability to determine which models to test:
       - Consumer hardware (no CUDA): Tests smaller models only (<6 GB)
       - Premium hardware (CUDA): Tests both regular and large models
    2. For each model, runs the answer analysis pipeline which:
       - Processes a handwritten answer image (via OCR)
       - Processes a spoken answer audio file (via ASR)
       - Evaluates answers against the provided text and question
    3. Logs execution time for each model to separate log files

Output:
    Creates log files in the logs/ directory with naming pattern:
        {model-name}-assessment.log

Usage:
    python -m src.scripts.timed_analysis_local
"""

import time

from pathlib import Path

import torch

from content_generation.vocabulary import default_text, default_questions
from assessment.analysis_local import analyse_answers
from content_generation.edu_content_local import LargeModel, Model
from scripts.utils import format_duration


CURRENT_MODULE_DIRPATH = Path(__file__).parent.resolve()
LOG_DIRPATH = CURRENT_MODULE_DIRPATH.parent.parent / "logs"


# If CUDA is available (premium hardware), evaluate both regular and large models.
# Otherwise (consumer-grade hardware), only evaluate regular models (< 6 GB)
if torch.cuda.is_available():
    print("CUDA is available. Evaluating both regular and large models.")
    models = list(Model) + list(LargeModel)
else:
    models = list(Model)


for model in models:
    print(f"Analysing an answer for model {model.value}...")
    log_filepath = LOG_DIRPATH / f"{model.value.replace('/', '-').replace(':', '-')}-assessment.log"
    start = time.perf_counter()

    analyse_answers(
        text=default_text,
        question=default_questions[0],
        handwritten_answer_filepath=CURRENT_MODULE_DIRPATH.parent / "assessment/model-answers/answer-0.png",
        spoken_answer_filepath=CURRENT_MODULE_DIRPATH.parent / "assessment/model-answers/answer-0.wav",
        model_name=model,
    )
    
    elapsed = time.perf_counter() - start
    with open(log_filepath, "a", encoding="utf-8") as f:
        f.write(f"Execution time: {format_duration(elapsed)}\n\n")

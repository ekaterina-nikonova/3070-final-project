"""
Timed question generation benchmarking (local models)
=====================================================

Purpose:
    Benchmarks local LLM models for generating comprehension questions from
    Japanese text. Tests each model with both long and short prompt variations
    to evaluate prompt engineering impact on generation quality and speed.

Workflow:
    1. Detects CUDA availability to determine available models:
       - Consumer hardware (no CUDA): Tests smaller models only (<6 GB)
       - Premium hardware (CUDA): Tests both regular and large models
    2. For each prompt style (long/short) and each model:
       - Generates comprehension questions for the default Japanese text
       - Measures and logs the execution time
    3. Creates separate log files for each model/prompt combination

Prompt Variations:
    - long-prompt: Detailed system message with comprehensive instructions
    - short-prompt: Concise system + user message pair

Output:
    Creates log files in logs/ directory with naming pattern:
        {prompt-style}{model-name}-questions.log

Usage:
    python -m src.scripts.timed_questions_generation_local
"""

import time

from pathlib import Path

import torch

from content_generation.edu_content_local import LargeModel, Model, generate_questions
from content_generation.prompt_utilities import make_questions_system_message, make_questions_system_message_short, make_questions_user_message_short
from content_generation.vocabulary import default_text
from scripts.utils import format_duration


CURRENT_MODULE_DIRPATH = Path(__file__).parent.resolve()
LOG_DIRPATH = CURRENT_MODULE_DIRPATH.parent.parent / "logs"


prompt_makers = {
    "long-prompt-": (make_questions_system_message, None),
    "short-prompt-": (make_questions_system_message_short, make_questions_user_message_short),
}

# If CUDA is available (premium hardware), evaluate both regular and large models.
# Otherwise (consumer-grade hardware), only evaluate regular models (< 6 GB)
if torch.cuda.is_available():
    print("CUDA is available. Evaluating both regular and large models.")
    models = list(Model) + list(LargeModel)
else:
    models = list(Model)


for prompt_prefix, (system_message_maker, user_message_maker) in prompt_makers.items():
    for model in models:
        print(f"Generating questions for model {model.value} with prompt prefix '{prompt_prefix}'...")
        log_filepath = LOG_DIRPATH / f"{prompt_prefix}{model.value.replace('/', '-').replace(':', '-')}-questions.log"
        start = time.perf_counter()
        generate_questions(default_text, model, system_message_maker, user_message_maker, log_filepath)
        elapsed = time.perf_counter() - start
        with open(log_filepath, "a", encoding="utf-8") as f:
            f.write(f"Execution time: {format_duration(elapsed)}\n\n")

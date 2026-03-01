"""
Timed question generation benchmarking (Perplexity API)
=======================================================

Purpose:
    Benchmarks the Perplexity API for generating comprehension questions from
    Japanese text. Tests both long and short prompt variations to evaluate
    prompt engineering impact when using cloud-based generation.

Workflow:
    1. For each prompt style (long/short):
       - Calls the Perplexity API to generate questions for default Japanese text
       - Measures and logs the execution time
    2. Creates separate log files for each prompt variation

Prompt Variations:
    - long-prompt: Detailed system message with comprehensive instructions
    - short-prompt: Concise system message

Output:
    Creates log files in logs/ directory:
        - long-prompt-perplexity-questions.log
        - short-prompt-perplexity-questions.log

Requirements:
    - PERPLEXITY_API_KEY environment variable must be set
    - Internet connection for API access

Usage:
    python -m src.scripts.timed_questions_generation_perplexity
"""

import time

from pathlib import Path

from content_generation.edu_content_perplexity import generate_questions
from content_generation.prompt_utilities import make_questions_system_message, make_questions_system_message_short
from content_generation.vocabulary import default_text
from scripts.utils import format_duration


CURRENT_MODULE_DIRPATH = Path(__file__).parent.resolve()
LOG_DIRPATH = CURRENT_MODULE_DIRPATH.parent.parent / "logs"


system_message_makers = {
    "long-prompt-": make_questions_system_message,
    "short-prompt-": make_questions_system_message_short,
}

for prompt_prefix, system_message_maker in system_message_makers.items():
    log_filepath = LOG_DIRPATH / f"{prompt_prefix}perplexity-questions.log"
    start = time.perf_counter()
    generate_questions(default_text, system_message_maker=system_message_maker, log_filepath=log_filepath)
    elapsed = time.perf_counter() - start
    with open(log_filepath, "a", encoding="utf-8") as f:
        f.write(f"Execution time: {format_duration(elapsed)}\n\n")

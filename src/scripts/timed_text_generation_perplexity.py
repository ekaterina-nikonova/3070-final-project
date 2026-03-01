"""
Timed text generation benchmarking (Perplexity API)
===================================================

Purpose:
    Benchmarks the Perplexity API for generating Japanese educational text content.
    Tests both long and short prompt variations to evaluate prompt engineering
    impact when using cloud-based text generation.

Workflow:
    1. For each prompt style (long/short):
       - Calls the Perplexity API to generate Japanese text on the default topic
       - Measures and logs the execution time
    2. Creates separate log files for each prompt variation

Prompt Variations:
    - long-prompt: Detailed system message with comprehensive content guidelines
    - short-prompt: Concise system + user message pair

Output:
    Creates log files in logs/ directory:
        - long-prompt-perplexity-text.log
        - short-prompt-perplexity-text.log

Requirements:
    - PERPLEXITY_API_KEY environment variable must be set
    - Internet connection for API access

Usage:
    python -m src.scripts.timed_text_generation_perplexity
"""

import time

from pathlib import Path

from content_generation.edu_content_perplexity import generate_text
from content_generation.prompt_utilities import make_text_system_message, make_text_system_message_short, make_text_user_message_short
from content_generation.vocabulary import default_topic
from scripts.utils import format_duration


CURRENT_MODULE_DIRPATH = Path(__file__).parent.resolve()
LOG_DIRPATH = CURRENT_MODULE_DIRPATH.parent.parent / "logs"


prompt_makers = {
    "long-prompt-": (make_text_system_message, None),
    "short-prompt-": (make_text_system_message_short, make_text_user_message_short),
}

for prompt_prefix, (system_message_maker, user_message_maker) in prompt_makers.items():
    log_filepath = LOG_DIRPATH / f"{prompt_prefix}perplexity-text.log"
    start = time.perf_counter()
    generate_text(
        default_topic,
        system_message_maker=system_message_maker,
        user_message_maker=user_message_maker,
        log_filepath=log_filepath)
    elapsed = time.perf_counter() - start
    with open(log_filepath, "a", encoding="utf-8") as f:
        f.write(f"Execution time: {format_duration(elapsed)}\n\n")

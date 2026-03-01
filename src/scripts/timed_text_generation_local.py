"""
Timed text generation benchmarking (local models)
=================================================

Purpose:
    Benchmarks local LLM models for generating Japanese educational text content
    on a given topic. Tests each model with both long and short prompt variations
    to evaluate generation quality and inference speed.

Workflow:
    1. Detects CUDA availability to determine available models:
       - Consumer hardware (no CUDA): Tests smaller models only (<6 GB)
       - Premium hardware (CUDA): Tests both regular and large models
    2. For each prompt style (long/short) and each model:
       - Generates Japanese text about the default topic
       - Measures and logs the execution time
    3. Creates separate log files for each model/prompt combination

Prompt Variations:
    - long-prompt: Detailed system message with comprehensive content guidelines
    - short-prompt: Concise system + user message pair

Output:
    Creates log files in logs/ directory with naming pattern:
        {prompt-style}{model-name}-text.log

Usage:
    python -m src.scripts.timed_text_generation_local
"""

import time

from pathlib import Path

import torch

from content_generation.edu_content_local import LargeModel, Model, generate_text
from content_generation.prompt_utilities import (
    make_text_system_message_short,
    make_text_user_message_short,
    make_text_system_message,
)
from content_generation.vocabulary import default_topic
from scripts.utils import format_duration

CURRENT_MODULE_DIRPATH = Path(__file__).parent.resolve()
LOG_DIRPATH = CURRENT_MODULE_DIRPATH.parent.parent / "logs"


prompt_makers = {
    "long-prompt-": (make_text_system_message, None),
    "short-prompt-": (make_text_system_message_short, make_text_user_message_short),
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
        print(f"Generating text for model {model.value} with prompt prefix '{prompt_prefix}'...")
        log_filepath = LOG_DIRPATH / f"{prompt_prefix}{model.value.replace('/', '-').replace(':', '-')}-text.log"
        start = time.perf_counter()
        generate_text(default_topic, model, system_message_maker, user_message_maker, log_filepath)
        elapsed = time.perf_counter() - start
        with open(log_filepath, "a", encoding="utf-8") as f:
            f.write(f"Execution time: {format_duration(elapsed)}\n\n")

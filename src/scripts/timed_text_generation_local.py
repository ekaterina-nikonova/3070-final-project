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

CURRENT_MODULE_DIRPATH = Path(__file__).parent.resolve()
LOG_DIRPATH = CURRENT_MODULE_DIRPATH.parent.parent / "logs"


def format_duration(seconds: float) -> str:
    ms = int((seconds - int(seconds)) * 1000)
    secs_total = int(seconds)
    hours, rem = divmod(secs_total, 3600)
    minutes, seconds_ = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_:02d}.{ms:03d}"

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

import time

from pathlib import Path

from content_generation.edu_content_perplexity import generate_text
from content_generation.prompt_utilities import make_text_system_message, make_text_system_message_short, make_text_user_message_short
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

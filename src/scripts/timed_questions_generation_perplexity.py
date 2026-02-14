import time

from pathlib import Path

from content_generation.edu_content_perplexity import generate_questions
from content_generation.prompt_utilities import make_questions_system_message, make_questions_system_message_short
from content_generation.vocabulary import default_text


CURRENT_MODULE_DIRPATH = Path(__file__).parent.resolve()
LOG_DIRPATH = CURRENT_MODULE_DIRPATH.parent.parent / "logs"


def format_duration(seconds: float) -> str:
    ms = int((seconds - int(seconds)) * 1000)
    secs_total = int(seconds)
    hours, rem = divmod(secs_total, 3600)
    minutes, seconds_ = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_:02d}.{ms:03d}"


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

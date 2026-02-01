import time

from pathlib import Path

from content_generation.edu_content import Model, generate_text
from content_generation.vocabulary import default_topic

CURRENT_MODULE_DIRPATH = Path(__file__).parent.resolve()
LOG_DIRPATH = CURRENT_MODULE_DIRPATH.parent.parent / "logs"


def format_duration(seconds: float) -> str:
    ms = int((seconds - int(seconds)) * 1000)
    secs_total = int(seconds)
    hours, rem = divmod(secs_total, 3600)
    minutes, seconds_ = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_:02d}.{ms:03d}"


for model in Model:
    log_filepath = LOG_DIRPATH / f"{model.value.replace('/', '-').replace(':', '-')}-text.log"
    start = time.perf_counter()
    generate_text(default_topic, model, log_filepath)
    elapsed = time.perf_counter() - start
    with open(log_filepath, "a", encoding="utf-8") as f:
        f.write(f"Execution time: {format_duration(elapsed)}\n\n")

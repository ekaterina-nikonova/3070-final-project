import time

from content_generation.edu_content import Model, generate_questions
from content_generation.vocabulary import default_text


def format_duration(seconds: float) -> str:
    ms = int((seconds - int(seconds)) * 1000)
    secs_total = int(seconds)
    hours, rem = divmod(secs_total, 3600)
    minutes, seconds_ = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_:02d}.{ms:03d}"


for model in Model:
    log_filepath = f"../../logs/{model.value.replace('/', '-').replace(':', '-')}-questions.log"
    start = time.perf_counter()
    generate_questions(default_text, model, log_filepath)
    elapsed = time.perf_counter() - start
    with open(log_filepath, "a", encoding="utf-8") as f:
        f.write(f"Execution time: {format_duration(elapsed)}\n\n")

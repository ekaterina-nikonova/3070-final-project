import time

from pathlib import Path

import torch

from content_generation.vocabulary import default_text, default_questions
from assessment.analysis_local import analyse_answers
from content_generation.edu_content_local import LargeModel, Model


CURRENT_MODULE_DIRPATH = Path(__file__).parent.resolve()
LOG_DIRPATH = CURRENT_MODULE_DIRPATH.parent.parent / "logs"


def format_duration(seconds: float) -> str:
    ms = int((seconds - int(seconds)) * 1000)
    secs_total = int(seconds)
    hours, rem = divmod(secs_total, 3600)
    minutes, seconds_ = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_:02d}.{ms:03d}"


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

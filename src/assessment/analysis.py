from pathlib import Path
from typing import Optional

from langchain_ollama import ChatOllama

from answer_processing.ocr import convert_to_text as image_to_text
from answer_processing.asr import convert_to_text as audio_to_text
from content_generation.edu_content import LargeModel, Model, DEFAULT_MODEL


CURRENT_MODULE_DIRPATH = Path(__file__).parent.resolve()
LOG_DIRPATH = CURRENT_MODULE_DIRPATH.parent.parent / "logs"


def analyse_answers(
    text: str, 
    question: str,
    handwritten_answer_filepath,
    spoken_answer_filepath,
    model_name: Model | LargeModel = DEFAULT_MODEL,
    log_filepath: Optional[Path] = None,
) -> str:
    """Analyse the answers using the provided text, question, handwritten answer image, and spoken answer audio.

    Args:
        text (str): The reference text.
        question (str): The question to be answered.
        handwritten_answer_filepath (str): Filepath to the handwritten answer image.
        spoken_answer_filepath (str): Filepath to the spoken answer audio.
        model_name (Model): The model to use for feedback generation.
        log_filepath (Optional[Path]): The path to the log file where feedback is written.
    Raises:
        GenerationError: If the feedback generation fails.

    Returns:
        str: Feedback on the student's answers.
    """

    if not text:
        raise ValueError("Text must be provided.")
    
    if not question:
        raise ValueError("Question must be specified.")

    if log_filepath is None:
        log_filepath = LOG_DIRPATH / f"{model_name.value.replace('/', '-').replace(':', '-')}-assessment.log"

    handwritten_answer = image_to_text(handwritten_answer_filepath)
    spoken_answer = audio_to_text(spoken_answer_filepath)

    system_message = (
        "You are a Japanese language teacher giving feedback on the A1-level student's answers.\n\n"
        "The student is answering a comprehension question for a text in two forms: handwritten and spoken. "
        "These two answers should have the same content.\n\n"
        f"Here is the reference text:\n{text}\n\n"
        f"The question is:\n{question}\n\n"
        "Give your feedback on the student's answers based on the text and the question. "
        "Keep in mind that the handwritten answer was processed using OCR and the spoken answer was processed using ASR, "
        "so there may be some recognition errors. Focus on the content and language use in your feedback.\n\n"
        "Provide constructive criticism and suggestions for improvement in a supportive manner. Respond in English."
    )

    user_message = (
        "The handwritten answer is:\n"
        f"{handwritten_answer}\n\n"
        "The spoken answer is:\n"
        f"{spoken_answer}\n\n"
    )

    model = ChatOllama(model=model_name, validate_model_on_init=True)
    
    messages = [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_message,
        }
    ]

    response = model.invoke([
        ("system", messages[0]["content"]),
        ("user", messages[1]["content"].format(text=text, question=question))
    ])

    with open(log_filepath, "a", encoding="utf-8") as log_f:
        log_f.write(f"Response for text:\n\n{text}\n\n")
        log_f.write(f"Question:\n\n{question}\n\n")
        log_f.write(f"System message:\n{system_message}\n\n")
        log_f.write(f"User message:\n{user_message}\n\n")
        log_f.write(f"{response}\n\n")

    return response.content

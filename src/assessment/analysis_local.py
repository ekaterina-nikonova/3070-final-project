from pathlib import Path
from typing import Optional

from langchain_ollama import ChatOllama

from answer_processing.ocr import convert_to_text as image_to_text
from answer_processing.asr import convert_to_text as audio_to_text
from content_generation.edu_content_local import (
    LargeModel,
    Model,
    DEFAULT_ASSESSMENT_MODEL,
    _strip_think_tags,
)


# The path to the logs directory in the project's root relative to this module: ../../logs
# This allows storing logs in the same location regardless of where this module is called from.
CURRENT_MODULE_DIRPATH = Path(__file__).parent.resolve()
LOG_DIRPATH = CURRENT_MODULE_DIRPATH.parent.parent / "logs"


def analyse_answers(
    text: str, 
    question: str,
    handwritten_answer_filepath,
    spoken_answer_filepath,
    model_name: Model | LargeModel = DEFAULT_ASSESSMENT_MODEL,
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

    # Validate that the reference text is provided (required for context).
    if not text:
        raise ValueError("Text must be provided.")
    
    # Validate that the selected question is provided (required for assessment).
    if not question:
        raise ValueError("Question must be specified.")

    # If no log filepath specified, create a default one using the name of the LLM.
    # Replace '/' and ':' characters in the model name with '-' to create a valid filename
    # on all platforms.
    if log_filepath is None:
        log_filepath = LOG_DIRPATH / f"{model_name.value.replace('/', '-').replace(':', '-')}-assessment.log"

    # Convert the handwritten answer image to text using OCR.
    handwritten_answer = image_to_text(handwritten_answer_filepath)
    # Convert the spoken answer audio to text using ASR.
    spoken_answer = audio_to_text(spoken_answer_filepath)

    # Construct the system message that defines the AI's role and provides context.
    # This includes the reference text and question, along with instructions for feedback.
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

    # Construct the user message containing the user's actual answers
    # in the form of the OCR and ASR transcriptions of the handwritten and spoken responses.
    user_message = (
        "The handwritten answer is:\n"
        f"{handwritten_answer}\n\n"
        "The spoken answer is:\n"
        f"{spoken_answer}\n\n"
    )

    # Initialize the Ollama chat model with the specified model name
    # (ensure the model exists before proceeding).
    model = ChatOllama(model=model_name, validate_model_on_init=True)
    
    # Send the messages to the model and get the response
    response = model.invoke([
        ("system", system_message),
        ("user", user_message)
    ])

    # Log the interaction to file for debugging and record-keeping.
    # Opens file in append mode to preserve previous logs.
    with open(log_filepath, "a", encoding="utf-8") as log_f:
        log_f.write(f"Response for text:\n\n{text}\n\n")
        log_f.write(f"Question:\n\n{question}\n\n")
        log_f.write(f"System message:\n{system_message}\n\n")
        log_f.write(f"User message:\n{user_message}\n\n")
        log_f.write(f"{response}\n\n")

    # Return the response content after removing any <think> tags
    # included in the response by some "thinking" models, such as DeepSeek.
    return _strip_think_tags(response.content)

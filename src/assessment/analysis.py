import json
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub.errors import GenerationError
from perplexity import Perplexity

from assessment.conversion import image_to_text, audio_to_text

load_dotenv()


CURRENT_MODULE_DIRPATH = Path(__file__).parent.resolve()
LOG_DIRPATH = CURRENT_MODULE_DIRPATH.parent.parent / "logs"


def analyse_answers(text, question, handwritten_answer_filepath, spoken_answer_filepath) -> str:
    """Analyse the answers using the provided text, question, handwritten answer image, and spoken answer audio.

    Args:
        text (str): The reference text.
        question (str): The question to be answered.
        handwritten_answer_filepath (str): Filepath to the handwritten answer image.
        spoken_answer_filepath (str): Filepath to the spoken answer audio.

    Raises:
        GenerationError: If the feedback generation fails.

    Returns:
        str: Feedback on the student's answers.
    """
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

    client = Perplexity()  # Automatically uses PERPLEXITY_API_KEY

    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        model="sonar",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "feedback": {
                            "type": "string",
                        }
                    },
                    "required": ["feedback"]
                }
            }
        }
    )

    try:
        feedback = json.loads(completion.choices[0].message.content)["feedback"]
    except KeyError:
        raise GenerationError("Failed to generate feedback. Check the logs for more details.")
    except json.decoder.JSONDecodeError:
        with open(LOG_DIRPATH / "perplexity/feedback.log", "a") as log_f:
            log_f.write(f"Failed to generate feedback for answers:\nHandwritten: {handwritten_answer!r}\nSpoken: {spoken_answer!r}\n\n")
            log_f.write(f"{completion}\n\n")
        raise GenerationError("Failed to parse feedback JSON. Check the logs for more details.")
    else:
        with open(LOG_DIRPATH / "perplexity/feedback.log", "a") as log_f:
            log_f.write(f"Generated feedback for answers:\nHandwritten: {handwritten_answer!r}\nSpoken: {spoken_answer!r}\n\n")
            log_f.write(f"{completion}\n\n")
        return feedback

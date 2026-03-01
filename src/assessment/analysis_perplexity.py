import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub.errors import GenerationError
from perplexity import Perplexity

from answer_processing.ocr import convert_to_text as image_to_text
from answer_processing.asr import convert_to_text as audio_to_text


# Load environment variables from .env file (must include PERPLEXITY_API_KEY)
load_dotenv()


# The path to the logs directory in the project's root relative to this module: ../../logs
# This allows storing logs in the same location regardless of where this module is called from.
CURRENT_MODULE_DIRPATH = Path(__file__).parent.resolve()
LOG_DIRPATH = CURRENT_MODULE_DIRPATH.parent.parent / "logs"


def analyse_answers(
    text,
    question, 
    handwritten_answer_filepath, 
    spoken_answer_filepath,
    log_filepath: Optional[Path] = None,
) -> str:
    """Analyse the answers using the provided text, question, handwritten answer image, and spoken answer audio.

    Args:
        text (str): The reference text.
        question (str): The question to be answered.
        handwritten_answer_filepath (str): Filepath to the handwritten answer image.
        spoken_answer_filepath (str): Filepath to the spoken answer audio.
        log_filepath (Optional[Path]): The path to the log file where feedback is written.

    Raises:
        GenerationError: If the feedback generation fails.

    Returns:
        str: Feedback on the student's answers.
    """
    # Set the default log filepath if not provided.
    if log_filepath is None:
        log_filepath = LOG_DIRPATH / "perplexity-assessment.log"

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

    # Initialize the Perplexity API client
    # (uses PERPLEXITY_API_KEY from the environment by default)
    client = Perplexity()  # Can specify api_key=... if not using environment variable

    # Send the chat completion request to the Perplexity API
    # using the "sonar" model with a JSON schema response format.
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
        # Define the expected JSON response structure.
        # This ensures the API returns a properly formatted JSON object with a "feedback" field
        # that can be parsed and extracted from the response content.
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

    # Try to extract the feedback from the API response.
    try:
        # Parse the JSON response and extract the "feedback" field.
        feedback = json.loads(completion.choices[0].message.content)["feedback"]
    except KeyError:
        # Handle cases where the "feedback" key is missing from the response.
        raise GenerationError("Failed to generate feedback. Check the logs for more details.")
    except json.decoder.JSONDecodeError:
        # Handle cases where the response is not a valid JSON.
        with open(log_filepath, "a", encoding="utf-8") as log_f:
            log_f.write(
                f"Failed to generate feedback for answers:\nHandwritten: {handwritten_answer!r}\nSpoken: {spoken_answer!r}\n\n")
            log_f.write(f"{completion}\n\n")
        raise GenerationError("Failed to parse feedback JSON. Check the logs for more details.")
    else:
        # Success case: log the successful generation and return the feedback.
        with open(log_filepath, "a", encoding="utf-8") as log_f:
            log_f.write(f"Generated feedback for answers:\nHandwritten: {handwritten_answer!r}\nSpoken: {spoken_answer!r}\n\n")
            # Log the full completion object for reference and debugging (includes the raw response content and metadata).
            log_f.write(f"{completion}\n\n")
        
        # Return the extracted feedback string
        return feedback

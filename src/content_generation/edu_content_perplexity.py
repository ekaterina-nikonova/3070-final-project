import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub.errors import GenerationError
from perplexity import Perplexity

from content_generation.prompt_utilities import make_text_system_message, make_questions_system_message

# Load environment variables from .env file (required for API keys)
load_dotenv()

# The path to the logs directory in the project's root relative to this module: ../../logs
# This allows storing logs in the same location regardless of where this module is called from.
CURRENT_MODULE_DIRPATH = Path(__file__).parent.resolve()
LOG_DIRPATH = CURRENT_MODULE_DIRPATH.parent.parent / "logs"


# Perplexity's lightweight, cost-efficient "sonar" model for generation.
# (https://docs.perplexity.ai/docs/getting-started/models/models/sonar)
DEFAULT_PERPLEXITY_MODEL = "sonar"


def generate_text(
    topic: str,
    system_message_maker: callable = make_text_system_message,
    user_message_maker: Optional[callable] = None,
    log_filepath: Optional[Path] = None,
) -> str:
    """Generates a text on the given topic using the Perplexity API.
    Args:
        topic: The topic to generate text about.
        system_message_maker: A function that takes the topic as input and returns the system message content.
        user_message_maker: An optional function that takes the topic as input and returns the user message content.
                            If not provided, a default passthrough function will be used.
        log_filepath: An optional Path to a log file where generation details will be recorded. 
                      If not provided, defaults to "logs/perplexity-text.log".
    Returns:
        The generated text as a string.
    Raises:
        ValueError: If the topic is not provided.
        GenerationError: If the API fails to generate a text.
    """
    # Set default log file path if not provided.
    if log_filepath is None:
        log_filepath = LOG_DIRPATH / "perplexity-text.log"
    
    # Validate that a topic was provided.
    if not topic:
        raise ValueError("Topic must be specified.")
    
    # If no user message maker is provided, use a simple passthrough function.
    if user_message_maker is None:
        # When the system message already contains all instructions,
        # the user message maker should simply return the topic.
        user_message_maker = lambda x: x  

    # Initialise the Perplexity API client.
    client = Perplexity()  # Automatically uses PERPLEXITY_API_KEY for authentication

    # Create a chat completion request with structured JSON output
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_message_maker(topic),
            },
            {
                "role": "user",
                "content": user_message_maker(topic),
            },
        ],
        # Choose a model -- supported models listed here: https://docs.perplexity.ai/docs/agent-api/models
        model=DEFAULT_PERPLEXITY_MODEL,
        # Request structured JSON response with a specific schema
        # to ensure the API returns the text in a predictable format.
        response_format={
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                        }
                    },
                    "required": ["text"]
                }
            }
        }
    )

    try:
        # Parse the JSON response and extract the "text" field.
        text = json.loads(completion.choices[0].message.content)["text"]
    except KeyError:
        # Raise error if the "text" field is missing from the response.
        raise GenerationError("Failed to generate a text.")
    except json.decoder.JSONDecodeError:
        # If JSON parsing fails, log the raw response for debugging.
        with open(log_filepath, "a", encoding="utf-8") as log_f:
            log_f.write(f"Failed to generate a text for topic: {topic!r}\n")
            log_f.write(f"{completion}\n\n")
    else:
        # On success, log the generated text and the messages used.
        with open(log_filepath, "a", encoding="utf-8") as log_f:
            log_f.write(f"Generated a text for topic: {topic!r}\n\n")
            log_f.write(f"System message:\n{system_message_maker(topic)}\n\n")
            log_f.write(f"User message:\n{user_message_maker(topic)}\n\n")
            log_f.write(f"{completion}\n\n")
        # Return the successfully generated text.
        return text


def generate_questions(
    text: str, 
    system_message_maker: callable = make_questions_system_message,
    user_message_maker: Optional[callable] = None,
    log_filepath: Optional[Path] = None,
) -> list[str]:
    # Set default log file path if not provided.
    if log_filepath is None:
        log_filepath = LOG_DIRPATH / "perplexity-questions.log"

    # Validate that the input text was provided.
    if not text:
        raise ValueError("Text must be provided.")
    
    # If no user message maker is provided, use a simple passthrough function.
    if user_message_maker is None:
        # When the system message already contains all instructions,
        # the user message maker should simply return the text.
        user_message_maker = lambda x: x

    # Initialize the Perplexity API client.
    client = Perplexity()  # Automatically uses PERPLEXITY_API_KEY for authentication

    # Create a chat completion request with structured JSON output
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_message_maker(),
            },
            {
                "role": "user",
                "content": user_message_maker(text),
            },
        ],
        # Choose a model -- supported models listed here: https://docs.perplexity.ai/docs/agent-api/models
        model=DEFAULT_PERPLEXITY_MODEL,
        # Request structured JSON response with an array of questions
        # to ensure the API returns questions in a predictable format.
        response_format={
            "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "questions": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                }
                            }
                        },
                        "required": ["questions"]
                    }
                }
        }
    )

    # Store the parsed questions
    questions = []
    try:
        # Parse the JSON response and extract the "questions" array.
        questions = json.loads(completion.choices[0].message.content)["questions"]
    except KeyError:
        # Raise error if the "questions" field is missing from the response.
        raise GenerationError("Failed to generate questions.")
    except json.decoder.JSONDecodeError:
        # If JSON parsing fails, log the raw response for debugging.
        with open(log_filepath, "a", encoding="utf-8") as log_f:
            log_f.write(f"Failed to generate questions.\n")
            log_f.write(f"{completion}\n\n")
    else:
        # On success, log the generated questions and the messages used.
        with open(log_filepath, "a", encoding="utf-8") as log_f:
            log_f.write(f"Generated questions for text:\n\n{text}\n\n")
            log_f.write(f"System message:\n{system_message_maker()}\n\n")
            log_f.write(f"User message:\n{user_message_maker(text)}\n\n")
            log_f.write(f"{completion}\n\n")
    # Return the list of generated questions (may be empty if parsing failed).
    return questions

import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub.errors import GenerationError
from perplexity import Perplexity

from content_generation.prompt_utilities import make_text_system_message, make_questions_system_message

load_dotenv()


CURRENT_MODULE_DIRPATH = Path(__file__).parent.resolve()
LOG_DIRPATH = CURRENT_MODULE_DIRPATH.parent.parent / "logs"


def generate_text(
    topic: str,
    system_message_maker: callable = make_text_system_message,
    user_message_maker: Optional[callable] = None,
    log_filepath: Optional[Path] = None,
) -> str:
    if log_filepath is None:
        log_filepath = LOG_DIRPATH / "perplexity-text.log"
    
    if not topic:
        raise ValueError("Topic must be specified.")
    
    if user_message_maker is None:
        # If the system message already contains instructions,
        # the user message maker will simply return the topic.
        user_message_maker = lambda x: x  

    client = Perplexity()  # Automatically uses PERPLEXITY_API_KEY

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
        model="sonar",
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
        text = json.loads(completion.choices[0].message.content)["text"]
    except KeyError:
        raise GenerationError("Failed to generate a text.")
    except json.decoder.JSONDecodeError:
        with open(log_filepath, "a", encoding="utf-8") as log_f:
            log_f.write(f"Failed to generate a text for topic: {topic!r}\n")
            log_f.write(f"{completion}\n\n")
    else:
        with open(log_filepath, "a", encoding="utf-8") as log_f:
            log_f.write(f"Generated a text for topic: {topic!r}\n\n")
            log_f.write(f"System message:\n{system_message_maker(topic)}\n\n")
            log_f.write(f"User message:\n{user_message_maker(topic)}\n\n")
            log_f.write(f"{completion}\n\n")
        return text


def generate_questions(
    text: str, 
    system_message_maker: callable = make_questions_system_message,
    user_message_maker: Optional[callable] = None,
    log_filepath: Optional[Path] = None,
) -> list[str]:
    if log_filepath is None:
        log_filepath = LOG_DIRPATH / "perplexity-questions.log"

    if not text:
        raise ValueError("Text must be provided.")
    
    if user_message_maker is None:
        # If the system message already contains instructions,
        # the user message maker will simply return the text.
        user_message_maker = lambda x: x

    client = Perplexity()  # Automatically uses PERPLEXITY_API_KEY

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
        model="sonar",
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

    questions = []
    try:
        questions = json.loads(completion.choices[0].message.content)["questions"]
    except KeyError:
        raise GenerationError("Failed to generate questions.")
    except json.decoder.JSONDecodeError:
        with open(log_filepath, "a", encoding="utf-8") as log_f:
            log_f.write(f"Failed to generate questions.\n")
            log_f.write(f"{completion}\n\n")
    else:
        with open(log_filepath, "a", encoding="utf-8") as log_f:
            log_f.write(f"Generated questions for text:\n\n{text}\n\n")
            log_f.write(f"System message:\n{system_message_maker()}\n\n")
            log_f.write(f"User message:\n{user_message_maker(text)}\n\n")
            log_f.write(f"{completion}\n\n")
    return questions

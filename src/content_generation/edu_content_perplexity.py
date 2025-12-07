import json

from dotenv import load_dotenv
from huggingface_hub.errors import GenerationError
from perplexity import Perplexity

from content_generation.prompt_utilities import make_text_system_message, make_questions_system_message

load_dotenv()


def generate_text(topic: str):
    if not topic:
        raise ValueError("Topic must be specified.")

    client = Perplexity()  # Automatically uses PERPLEXITY_API_KEY

    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": make_text_system_message(topic),
            },
            {
                "role": "user",
                "content": topic,
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
        with open("../../logs/perplexity/perplexity.log", "a") as log_f:
            log_f.write(f"Failed to generate a text for topic: {topic!r}\n")
            log_f.write(f"{completion}\n\n")
    else:
        with open("../../logs/perplexity/perplexity.log", "a") as log_f:
            log_f.write(f"Generated a text for topic: {topic!r}\n")
            log_f.write(f"{completion}\n\n")
        return text


def generate_questions(text: str) -> list[str]:
    if not text:
        raise ValueError("Topic must be provided.")

    client = Perplexity()  # Automatically uses PERPLEXITY_API_KEY

    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": make_questions_system_message(),
            },
            {
                "role": "user",
                "content": text,
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
        with open("../../logs/perplexity/perplexity.log", "a") as log_f:
            log_f.write(f"Failed to generate questions.\n")
            log_f.write(f"{completion}\n\n")
    else:
        with open("../../logs/perplexity/perplexity.log", "a") as log_f:
            log_f.write(f"Generated questions for text:\n\n{text}\n\n")
            log_f.write(f"{completion}\n\n")
    return questions

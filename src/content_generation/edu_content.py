from enum import StrEnum
from typing import Optional

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from content_generation.prompt_utilities import (
    make_text_system_message_short,
    make_text_user_message_short,
    make_questions_system_message_short,
    make_questions_user_message_short,
)
from retrieval.embedding import fetch_similar_entries

load_dotenv()

class Model(StrEnum):
    GEMMA3_270M = "gemma3:270m"                         # 292 MB; 32K context
    GEMMA3_1B = "gemma3:1b"                             # 815 MB; 32K context
    GEMMA_JPN = "schroneko/gemma-2-2b-jpn-it:q4_K_S"    # 1.6 GB; 8K context
    DEEPSEEK_R1_8B = "deepseek-r1:8b"                   # 5.2 GB; 128K context
    QWEN3_4B = "qwen3:4b"                               # 2.5 GB; 256K context


def generate_text(
    topic: str,
    model_name: Model = Model.GEMMA_JPN,
    log_filepath: Optional[str] = None,
) -> str:
    """Generate text based on the specified topic using the specified model.

    Args:
        topic: The topic for which the text is to be generated.
        model_name: The name of the model to use for generating text. Must be supported by Ollama and installed.
        log_filepath: The path to the file where the generated text will be logged.

    Raises:
        ValueError: If a topic is not specified.

    Returns:
        The generated text.
    """
    if not topic:
        raise ValueError("Topic must be specified.")

    if log_filepath is None:
        log_filepath = f"../../logs/{model_name.value.replace('/', '-').replace(':', '-')}-text.log"

    model = ChatOllama(model=model_name, validate_model_on_init=True)

    vocabulary_sentences = fetch_similar_entries(topic, results_num=50, fetch_sentences=True)

    messages = [
        {
            "role": "system",
            "content": make_text_system_message_short(topic),
        },
        {
            "role": "user",
            "content": make_text_user_message_short(topic)
        }
    ]

    response = model.invoke([
        ("system", messages[0]["content"]),
        ("user", messages[1]["content"].format(topic=topic, sentences="\n".join(vocabulary_sentences)))
    ])

    with open(log_filepath, "a") as log_f:
        log_f.write(f"Response for topic:\n\n{topic}\n\n")
        log_f.write(f"{response}\n\n")

    return response.content


def generate_questions(
    text: str,
    model_name: Model = Model.GEMMA_JPN,
    log_filepath: Optional[str] = None,
) -> list[str]:
    """
    Generate questions based on the input text using the specified model.

    Args:
        text: The input text for which questions are to be generated.
        model_name: The name of the model to use for generating questions. Must be supported by Ollama and installed.
        log_filepath: The path to the file where the generated questions will be logged.

    Raises:
        ValueError: If text is not provided.

    Returns:
        A list of generated questions.
    """
    if not text:
        raise ValueError("Text must be provided.")

    if log_filepath is None:
        log_filepath = f"../../logs/{model_name.value.replace('/', '-').replace(':', '-')}-questions.log"

    model = ChatOllama(model=model_name, validate_model_on_init=True)

    messages = [
        {
            "role": "system",
            "content": make_questions_system_message_short(),
        },
        {
            "role": "user",
            "content": make_questions_user_message_short(text)
        }
    ]

    response = model.invoke([
        ("system", messages[0]["content"]),
        ("user", messages[1]["content"].format(text=text))
    ])

    with open(log_filepath, "a") as log_f:
        log_f.write(f"Response for text:\n\n{text}\n\n")
        log_f.write(f"{response}\n\n")

    return response.content

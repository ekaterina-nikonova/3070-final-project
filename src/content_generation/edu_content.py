import re
from enum import StrEnum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import torch

from content_generation.prompt_utilities import (
    make_questions_system_message_short,
    make_questions_user_message_short,
    make_text_system_message,
)
from retrieval.embedding import fetch_similar_entries


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from LLM responses (DeepSeek reasoning)."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def _parse_questions_from_response(response_text: str) -> list[str]:
    """
    Parse questions from the LLM response text.
    
    Handles:
    - <think>...</think> tags (DeepSeek reasoning)
    - Numbered lists (1. Question, 2. Question, etc.)
    - Line-separated questions
    """
    text = _strip_think_tags(response_text)
    
    # Split by common list patterns: "1.", "2.", etc. or newlines
    # First try numbered list pattern
    lines = re.split(r'\n+', text.strip())
    
    questions = []
    for line in lines:
        # Remove numbering like "1.", "2)", "1:" etc.
        cleaned = re.sub(r'^\s*\d+[\.\)\:]\s*', '', line.strip())
        # Only include non-empty lines that look like questions (contain Japanese or ?)
        if cleaned and (re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', cleaned) or '?' in cleaned):
            questions.append(cleaned)
    
    return questions

CURRENT_MODULE_DIRPATH = Path(__file__).parent.resolve()
LOG_DIRPATH = CURRENT_MODULE_DIRPATH.parent.parent / "logs"

load_dotenv()


class Model(StrEnum):
    GEMMA3_270M = "gemma3:270m"                         # 292 MB; 32K context
    GEMMA3_1B = "gemma3:1b"                             # 815 MB; 32K context
    GEMMA_JPN = "schroneko/gemma-2-2b-jpn-it:q4_K_S"    # 1.6 GB; 8K context
    DEEPSEEK_R1_8B = "deepseek-r1:8b"                   # 5.2 GB; 128K context
    QWEN3_4B = "qwen3:4b"                               # 2.5 GB; 256K context


class LargeModel(StrEnum):
    DEEPSEEK_R1_32B = "deepseek-r1:32b"                 # 20 GB; 128K context
    GEMMA3_27B = "gemma3:27b"                           # 17 GB; 128K context
    QWEN3_30B = "qwen3:30b"                             # 19 GB; 256K context
    YUMA_DEEPSEEK_JP_32_B = (
        "yuma/DeepSeek-R1-Distill-Qwen-Japanese:32b")   # 20 GB; 128K context


# Models that proved to be best-performing on consumer-grade and premium hardware
# according to the evaluation.
DEFAULT_TEXT_MODEL = (
    LargeModel.GEMMA3_27B
    if torch.cuda.is_available() else 
    Model.GEMMA_JPN
)

DEFAULT_QUESTION_MODEL = (
    LargeModel.GEMMA3_27B
    if torch.cuda.is_available() else 
    Model.GEMMA_JPN
)

DEFAULT_ASSESSMENT_MODEL = (
    LargeModel.YUMA_DEEPSEEK_JP_32_B
    if torch.cuda.is_available() else 
    Model.GEMMA_JPN
)

def generate_text(
    topic: str,
    model_name: Model = DEFAULT_TEXT_MODEL,
    system_message_maker: callable = make_text_system_message,
    user_message_maker: Optional[callable] = None,
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
        log_filepath = LOG_DIRPATH / f"{model_name.value.replace('/', '-').replace(':', '-')}-text.log"

    if user_message_maker is None:
        # If the system message already contains instructions,
        # the user message maker will simply return the topic.
        user_message_maker = lambda x: x

    model = ChatOllama(model=model_name, validate_model_on_init=True)

    vocabulary_sentences = fetch_similar_entries(topic, results_num=50, fetch_sentences=True)

    messages = [
        {
            "role": "system",
            "content": system_message_maker(topic),
        },
        {
            "role": "user",
            "content": user_message_maker(topic)
        }
    ]

    response = model.invoke([
        ("system", messages[0]["content"]),
        ("user", messages[1]["content"].format(topic=topic, sentences="\n".join(vocabulary_sentences)))
    ])

    with open(log_filepath, "a", encoding="utf-8") as log_f:
        log_f.write(f"Response for topic:\n\n{topic}\n\n")
        log_f.write(f"System message:\n{system_message_maker(topic)}\n\n")
        log_f.write(f"User message:\n{user_message_maker(topic)}\n\n")    
        log_f.write(f"{response}\n\n")

    return _strip_think_tags(response.content)


def generate_questions(
    text: str,
    model_name: Model = DEFAULT_QUESTION_MODEL,
    system_message_maker: callable = make_questions_system_message_short,
    user_message_maker: Optional[callable] = make_questions_user_message_short,
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
        log_filepath = LOG_DIRPATH / f"{model_name.value.replace('/', '-').replace(':', '-')}-questions.log"

    if user_message_maker is None:
        # If the system message already contains instructions,
        # the user message maker will simply return the text.
        user_message_maker = lambda x: x

    model = ChatOllama(model=model_name, validate_model_on_init=True)

    messages = [
        {
            "role": "system",
            "content": system_message_maker(),
        },
        {
            "role": "user",
            "content": user_message_maker(text)
        }
    ]

    response = model.invoke([
        ("system", messages[0]["content"]),
        ("user", messages[1]["content"].format(text=text))
    ])

    with open(log_filepath, "a", encoding="utf-8") as log_f:
        log_f.write(f"Response for text:\n\n{text}\n\n")
        log_f.write(f"System message:\n{system_message_maker()}\n\n")
        log_f.write(f"User message:\n{user_message_maker(text)}\n\n")
        log_f.write(f"{response}\n\n")

    return _parse_questions_from_response(response.content)

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fugashi import Tagger
from langchain_ollama import ChatOllama
import torch

from content_generation.prompt_utilities import (
    DEFAULT_TEXT_LENGTH_CHAR,
    make_questions_system_message_short,
    make_questions_user_message_short,
    make_text_system_message,
)
from retrieval.embedding import fetch_similar_entries


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from LLM responses (DeepSeek reasoning)."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


# Initialize the Japanese tokenizer (using MeCab via fugashi with unidic-lite dictionary)
_tagger = Tagger('-Owakati')


def _tokenize_japanese(text: str) -> list[str]:
    """Tokenize Japanese text into words using fugashi (MeCab wrapper)."""
    return [word.surface for word in _tagger(text)]


def _extract_vocabulary_set(vocabulary_sentences: list[str]) -> set[str]:
    """Extract all unique words from a list of vocabulary sentences."""
    words = set()
    for sentence in vocabulary_sentences:
        words.update(_tokenize_japanese(sentence))
    return words


@dataclass
class TextVerificationResult:
    """Result of verifying generated text against vocabulary constraints."""
    text: str
    is_valid: bool
    char_count: int
    extra_words: set[str]
    
    @property
    def violation_count(self) -> int:
        """Return the total number of violations (extra characters + extra words)."""
        char_overflow = max(0, self.char_count - DEFAULT_TEXT_LENGTH_CHAR)
        return char_overflow + len(self.extra_words)


def _verify_text(
    text: str,
    allowed_words: set[str],
    max_chars: int = DEFAULT_TEXT_LENGTH_CHAR,
) -> TextVerificationResult:
    """Verify that the generated text meets vocabulary and length constraints.
    
    Args:
        text: The generated Japanese text to verify.
        allowed_words: Set of allowed words from vocabulary sentences.
        max_chars: Maximum allowed character count.
    
    Returns:
        TextVerificationResult with validation details.
    """
    char_count = len(text)
    text_words = set(_tokenize_japanese(text))
    extra_words = text_words - allowed_words
    
    is_valid = char_count <= max_chars and len(extra_words) == 0
    
    return TextVerificationResult(
        text=text,
        is_valid=is_valid,
        char_count=char_count,
        extra_words=extra_words,
    )


def _make_rewrite_prompt(
    original_text: str,
    extra_words: set[str],
    vocabulary_sentences: list[str],
    max_chars: int,
) -> str:
    """Create a prompt asking the model to rewrite the text without extra words."""
    extra_words_str = ", ".join(extra_words)
    vocab_str = "\n".join(vocabulary_sentences)

    print("=== Rewrite Prompt ===")
    print(f"Making rewrite prompt with extra words: {extra_words_str}")
    print("Original text:")
    print(original_text)
    
    return (
        f"Please rewrite the following Japanese text.\n\n"
        f"Original text:\n{original_text}\n\n"
        f"Problem: The following words are NOT allowed: {extra_words_str}\n\n"
        f"Allowed vocabulary sentences:\n{vocab_str}\n\n"
        f"Constraints:\n"
        f"- Maximum {max_chars} characters\n"
        f"- Use ONLY words that appear in the vocabulary sentences above\n"
        f"- Convey the same meaning as the original\n\n"
        f"Output ONLY the rewritten Japanese text, nothing else."
    )


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

# Maximum number of verification/rewrite attempts for text generation
MAX_VERIFICATION_ATTEMPTS = 3


def generate_text(
    topic: str,
    model_name: Model = DEFAULT_TEXT_MODEL,
    system_message_maker: callable = make_text_system_message,
    user_message_maker: Optional[callable] = None,
    log_filepath: Optional[str] = None,
    max_attempts: int = MAX_VERIFICATION_ATTEMPTS,
) -> str:
    """Generate text based on the specified topic using the specified model.

    The function verifies that the generated text:
    1. Contains no more than DEFAULT_TEXT_LENGTH_CHAR characters
    2. Only uses words found in the vocabulary sentences
    
    If verification fails, the model is asked to rewrite the text up to max_attempts times.
    The best text (with fewest violations) is returned.

    Args:
        topic: The topic for which the text is to be generated.
        model_name: The name of the model to use for generating text. Must be supported by Ollama and installed.
        log_filepath: The path to the file where the generated text will be logged.
        max_attempts: Maximum number of verification/rewrite attempts (default: 3).

    Raises:
        ValueError: If a topic is not specified.

    Returns:
        The generated text (best result after verification loop).
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
    allowed_words = _extract_vocabulary_set(vocabulary_sentences)

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

    generated_text = _strip_think_tags(response.content)
    
    # Verification loop
    best_result: TextVerificationResult | None = None
    
    for attempt in range(max_attempts):
        verification = _verify_text(generated_text, allowed_words)
        
        # Log the attempt
        with open(log_filepath, "a", encoding="utf-8") as log_f:
            log_f.write(f"=== Attempt {attempt + 1}/{max_attempts} ===\n")
            log_f.write(f"Response for topic:\n\n{topic}\n\n")
            log_f.write(f"Generated text ({verification.char_count} chars):\n{generated_text}\n\n")
            if verification.extra_words:
                log_f.write(f"Extra words found: {verification.extra_words}\n\n")
            log_f.write(f"Valid: {verification.is_valid}\n\n")
        
        # Track the best result
        if best_result is None or verification.violation_count < best_result.violation_count:
            best_result = verification
        
        # If valid, return immediately
        if verification.is_valid:
            return generated_text
        
        # If not the last attempt, ask for a rewrite
        if attempt < max_attempts - 1:
            rewrite_prompt = _make_rewrite_prompt(
                original_text=generated_text,
                extra_words=verification.extra_words,
                vocabulary_sentences=vocabulary_sentences,
                max_chars=DEFAULT_TEXT_LENGTH_CHAR,
            )
            
            rewrite_response = model.invoke([
                ("system", "You are a Japanese teacher writing text for A1-level learners."),
                ("user", rewrite_prompt)
            ])
            
            generated_text = _strip_think_tags(rewrite_response.content)
    
    # Return the best result after all attempts
    with open(log_filepath, "a", encoding="utf-8") as log_f:
        log_f.write(f"=== Returning best result (violations: {best_result.violation_count}) ===\n\n")
    
    return best_result.text


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

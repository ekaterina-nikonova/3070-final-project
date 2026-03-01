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
    # Use regex to find and remove all <think>...</think> blocks (including content inside).
    # re.DOTALL makes '.' match newlines, so multi-line think blocks are removed.
    # .strip() removes any leading/trailing whitespace from the cleaned result.
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


# Initialise the Japanese tokenizer (using MeCab via fugashi with unidic-lite dictionary)
# '-Owakati' option configures the tagger for word-segmented (wakati-gaki) output
# (See more about MeCab options here (in Japanese): https://taku910.github.io/mecab/#wakati)
_tagger = Tagger('-Owakati')


def _tokenize_japanese(text: str) -> list[str]:
    """Tokenize Japanese text into words using fugashi (MeCab wrapper)."""
    # Pass text through the MeCab tagger and extract the surface form of each token.
    # word.surface contains the actual text of each morpheme (word segment)
    return [word.surface for word in _tagger(text)]


def _extract_vocabulary_set(vocabulary_sentences: list[str]) -> set[str]:
    """Extract all unique words from a list of vocabulary sentences."""
    # Initialise an empty set to store unique words.
    words = set()
    # Tokenise each vocabulary sentence into words and add them to the set
    for sentence in vocabulary_sentences:
        words.update(_tokenize_japanese(sentence))  # automatically handles deduplication
    return words


@dataclass
class TextVerificationResult:
    """Result of verifying generated text against vocabulary constraints."""
    
    text: str              # the generated text
    is_valid: bool         # whether the text passes all validation checks (length and vocabulary)
    char_count: int        # total number of characters in the text
    extra_words: set[str]  # set of words found in text that are not in the allowed vocabulary
    
    @property
    def violation_count(self) -> int:
        """The total number of violations (extra characters + extra words)."""
        # Calculate how many characters exceed the maximum allowed length.
        # If the text length is under the limit, this will be 0 due to max(0, ...).
        char_overflow = max(0, self.char_count - DEFAULT_TEXT_LENGTH_CHAR)
        # Total violations = excess characters + number of disallowed words
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
    # Count the total number of characters in the generated text.
    char_count = len(text)
    # Tokenise the text and convert it into a set for efficient comparison.
    text_words = set(_tokenize_japanese(text))
    # Find words in the text that are not in the allowed vocabulary
    extra_words = text_words - allowed_words
    
    # The text is valid only if it's within the character limit and uses no extra words.
    is_valid = char_count <= max_chars and len(extra_words) == 0
    
    # Return a structured result containing verification details.
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
    # Convert the set of disallowed words to a comma-separated string for display.
    extra_words_str = ", ".join(extra_words)
    # Join vocabulary sentences with newlines for readable display in the prompt.
    vocab_str = "\n".join(vocabulary_sentences)
    
    # Construct a detailed rewrite prompt with constraints for the model.
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
    """Parse questions from the LLM response text as a list of strings.
    
    Handles:
    - <think>...</think> tags (DeepSeek reasoning)
    - Numbered lists (1. Question, 2. Question, etc.)
    - Line-separated questions
    """
    # Strip DeepSeek-style reasoning blocks from the response
    text = _strip_think_tags(response_text)
    
    # Split by common list patterns: "1.", "2.", etc. or newlines
    lines = re.split(r'\n+', text.strip())  # extract one question per line, ignoring empty lines    
    questions = []  # extracted questions
    for line in lines:
        # Remove numbering like "1.", "2)", "1:" etc. from the start of each line.
        # The regex matches optional whitespace, digits, and common list markers.
        cleaned = re.sub(r'^\s*\d+[\.\)\:]\s*', '', line.strip())
        # Only include non-empty lines that look like questions (contain Japanese or ?):
        # check for Japanese characters (Hiragana, Katakana, Kanji) or question marks
        if cleaned and (re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', cleaned) or '?' in cleaned):
            questions.append(cleaned)
    
    # Return the list of parsed questions
    return questions


# The path to the logs directory in the project's root relative to this module: ../../logs
# This allows storing logs in the same location regardless of where this module is called from.
CURRENT_MODULE_DIRPATH = Path(__file__).parent.resolve()
LOG_DIRPATH = CURRENT_MODULE_DIRPATH.parent.parent / "logs"

# Load environment variables from .env file (required for API keys)
load_dotenv()


# Enumeration of small/medium models suitable for consumer-grade hardware
# with the model size and its context window length.
class Model(StrEnum):
    GEMMA3_270M = "gemma3:270m"                         # 292 MB; 32K context
    GEMMA3_1B = "gemma3:1b"                             # 815 MB; 32K context
    GEMMA_JPN = "schroneko/gemma-2-2b-jpn-it:q4_K_S"    # 1.6 GB; 8K context
    DEEPSEEK_R1_8B = "deepseek-r1:8b"                   # 5.2 GB; 128K context
    QWEN3_4B = "qwen3:4b"                               # 2.5 GB; 256K context


# Enumeration of large models requiring premium hardware (high VRAM GPUs).
# IMPORTANT: Leave enough headroom in GPU memory for the OS tasks to avoid out-of-memory errors.
class LargeModel(StrEnum):
    DEEPSEEK_R1_32B = "deepseek-r1:32b"                 # 20 GB; 128K context
    GEMMA3_27B = "gemma3:27b"                           # 17 GB; 128K context
    QWEN3_30B = "qwen3:30b"                             # 19 GB; 256K context
    YUMA_DEEPSEEK_JP_32_B = (
        "yuma/DeepSeek-R1-Distill-Qwen-Japanese:32b")   # 20 GB; 128K context


# Models that proved to be best-performing on consumer-grade and premium hardware
# according to the evaluation.
# Select larger model if CUDA GPU is available, otherwise use smaller Japanese model
DEFAULT_TEXT_MODEL = (
    LargeModel.GEMMA3_27B
    if torch.cuda.is_available() else 
    Model.GEMMA_JPN
)

# Default model for generating comprehension questions
DEFAULT_QUESTION_MODEL = (
    LargeModel.GEMMA3_27B
    if torch.cuda.is_available() else 
    Model.GEMMA_JPN
)

# Default model for assessing the user's answers
DEFAULT_ASSESSMENT_MODEL = (
    LargeModel.YUMA_DEEPSEEK_JP_32_B
    if torch.cuda.is_available() else 
    Model.GEMMA_JPN
)

# Maximum number of verification/rewrite attempts for text generation.
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
        system_message_maker: A function that takes the topic as input and returns the system message string for the LLM.
        user_message_maker: A function that takes the topic as input and returns the user message string for the LLM. 
                            If None, a simple passthrough function will be used that returns the topic.
        log_filepath: The path to the file where the generated text will be logged.
        max_attempts: Maximum number of verification/rewrite attempts (default: 3).

    Raises:
        ValueError: If a topic is not specified.

    Returns:
        The generated text (best result after verification loop).
    """
    # Validate that a topic was provided.
    if not topic:
        raise ValueError("Topic must be specified.")

    # Set default log file path if not provided.
    # Replace special characters in model name to create a valid filename on any platform.
    if log_filepath is None:
        log_filepath = LOG_DIRPATH / f"{model_name.value.replace('/', '-').replace(':', '-')}-text.log"

    # If no user message maker is provided, use a simple passthrough function.
    if user_message_maker is None:
        # When the system message already contains all instructions,
        # the user message maker should simply return the topic.
        user_message_maker = lambda x: x

    # Initialise the Ollama LLM client with the specified model ensuring it is available in Ollama.
    model = ChatOllama(model=model_name, validate_model_on_init=True)

    # Fetch vocabulary sentences semantically similar to the topic from the vector store.
    # These sentences define the allowed vocabulary for the generated text.
    vocabulary_sentences = fetch_similar_entries(topic, results_num=50, fetch_sentences=True)
    # Extract all unique words from vocabulary sentences into a set for validation.
    allowed_words = _extract_vocabulary_set(vocabulary_sentences)

    # Send the initial request to the LLM and get the response.
    response = model.invoke([
        ("system", system_message_maker(topic)),
        ("user", user_message_maker(topic))
    ])

    # Remove any reasoning blocks from the response (for DeepSeek models in the "thinking" mode).
    generated_text = _strip_think_tags(response.content)
    
    # Verification loop: track the best result across multiple attempts.
    # The best result will be updated with the least-violating result.
    best_result: TextVerificationResult | None = None
    
    # Iterate through verification attempts
    for attempt in range(max_attempts):
        # Verify the generated text against vocabulary and length constraints.
        verification = _verify_text(generated_text, allowed_words)
        
        # Log the attempt details to the log file for debugging.
        with open(log_filepath, "a", encoding="utf-8") as log_f:
            log_f.write(f"=== Attempt {attempt + 1}/{max_attempts} ===\n")
            log_f.write(f"Response for topic:\n\n{topic}\n\n")
            log_f.write(f"Generated text ({verification.char_count} chars):\n{generated_text}\n\n")
            # Only log extra words if there are any violations.
            if verification.extra_words:
                log_f.write(f"Extra words found: {verification.extra_words}\n\n")
            log_f.write(f"Valid: {verification.is_valid}\n\n")
        
        # Track the best result and update if this is the first result or has fewer violations.
        if best_result is None or verification.violation_count < best_result.violation_count:
            best_result = verification
        
        # If the text passes all validation checks, return it immediately...
        if verification.is_valid:
            return generated_text
        
        # ...otherwise request a rewrite from the model if there are remaining attempts.
        if attempt < max_attempts - 1:
            # Create a prompt that instructs the model to fix the violations
            rewrite_prompt = _make_rewrite_prompt(
                original_text=generated_text,
                extra_words=verification.extra_words,
                vocabulary_sentences=vocabulary_sentences,
                max_chars=DEFAULT_TEXT_LENGTH_CHAR,
            )
            
            # Send the rewrite request to the model
            rewrite_response = model.invoke([
                ("system", "You are a Japanese teacher writing a text for A1-level learners."),
                ("user", rewrite_prompt)
            ])
            
            # Extract the rewritten text by stripping any reasoning blocks.
            generated_text = _strip_think_tags(rewrite_response.content)
    
    # After all attempts are exhausted, log and return the best result so far.
    with open(log_filepath, "a", encoding="utf-8") as log_f:
        log_f.write(f"=== Returning best result (violations: {best_result.violation_count}) ===\n\n")
    
    # Return the text with the fewest violations.
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
        system_message_maker: A function that takes no input and returns the system message string for the LLM.
        user_message_maker: A function that takes the input text and returns the user message string for the LLM. 
                            If None, a simple passthrough function will be used that returns the input text.
        log_filepath: The path to the file where the generated questions will be logged.

    Raises:
        ValueError: If text is not provided.

    Returns:
        A list of generated questions.
    """
    # Validate that the input text was provided.
    if not text:
        raise ValueError("Text must be provided.")

    # Set default log file path if not provided.
    # Replace special characters in model name to create a valid filename on any platform.
    if log_filepath is None:
        log_filepath = LOG_DIRPATH / f"{model_name.value.replace('/', '-').replace(':', '-')}-questions.log"

    # If no user message maker is provided, use a simple passthrough function.
    if user_message_maker is None:
        # When the system message already contains all instructions,
        # the user message maker should simply return the text.
        user_message_maker = lambda x: x

    # Initialise the Ollama LLM client with the specified model ensuring it is available in Ollama.
    model = ChatOllama(model=model_name, validate_model_on_init=True)

    # Send the request to the LLM and get the response
    response = model.invoke([
        ("system", system_message_maker()),
        ("user", user_message_maker(text))
    ])

    # Log the request and response details for debugging.
    with open(log_filepath, "a", encoding="utf-8") as log_f:
        log_f.write(f"Response for text:\n\n{text}\n\n")
        log_f.write(f"System message:\n{system_message_maker()}\n\n")
        log_f.write(f"User message:\n{user_message_maker(text)}\n\n")
        log_f.write(f"{response}\n\n")

    # Parse the response content to extract individual questions.
    return _parse_questions_from_response(response.content)

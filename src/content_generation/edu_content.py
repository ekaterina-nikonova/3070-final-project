from enum import StrEnum

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from content_generation.prompt_utilities import make_text_system_message, make_questions_system_message
from retrieval.embedding import fetch_similar_entries

load_dotenv()

class Model(StrEnum):
    GEMMA3_270M = "gemma3:270m"
    GEMMA3_1B = "gemma3:1b"
    GEMMA_JPN = "schroneko/gemma-2-2b-jpn-it:q4_K_S"
    DEEPSEEK_R1_8B = "deepseek-r1:8b"
    QWEN3_4B = "qwen3:4b"


def generate_text(topic: str, model_name: Model = Model.GEMMA_JPN) -> str:
    if not topic:
        raise ValueError("Topic must be specified.")

    model = ChatOllama(model=model_name, validate_model_on_init=True)

    vocabulary_sentences = fetch_similar_entries(topic, results_num=50, fetch_sentences=True)

    messages = [
        {
            "role": "system",
            "content": make_text_system_message(topic),
        },
        {
            "role": "user",
            "content": topic
        }
    ]

    response = model.invoke([
        ("system", messages[0]["content"]),
        ("user", messages[1]["content"].format(topic=topic, sentences="\n".join(vocabulary_sentences)))
    ])

    text = response.choices[0].message.content
    with open(f"../../logs/{model_name}-text.log", "a") as log_f:
        log_f.write(f"Response for topic:\n\n{topic}\n\n")
        log_f.write(f"{response}\n\n")

    return text


def generate_questions(text: str, model_name: Model = Model.GEMMA_JPN) -> list[str]:
    if not text:
        raise ValueError("Text must be provided.")

    print("Text:\n", text)

    model = ChatOllama(model=model_name, validate_model_on_init=True)

    messages = [
        {
            "role": "system",
            "content": make_questions_system_message(),
        },
        {
            "role": "user",
            "content": text
        }
    ]

    response = model.invoke([
        ("system", messages[0]["content"]),
        ("user", messages[1]["content"].format(text=text))
    ])

    questions = response.choices[0].message.content
    with open(f"../../logs/{model_name}-questions.log", "a") as log_f:
        log_f.write(f"Response for text:\n\n{text}\n\n")
        log_f.write(f"{response}\n\n")

    return questions

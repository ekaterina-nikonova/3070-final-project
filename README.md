# Japanese Language Exercise System (JES)

A web-based educational tool for Japanese language learning. The system generates reading passages and comprehension questions using LLMs and the RAG technique accessing a vector store containing the vocabulary to be trained. Handwritten and spoken answers are collected from the learner, and provides feedback on their grammatical correctness and appropriateness.

## Prerequisites

### uv (Python package manager)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) for fast dependency management.

### API Keys

Create a `.env` file in the project root with the following variables:

```
PERPLEXITY_API_KEY=your_perplexity_api_key_here
```

### Ollama (for local models)

For local text and question generation, install [Ollama](https://ollama.com/download) on your machine. Pre-download the models to avoid delays during first execution:

```shell
ollama pull gemma3:270m
ollama pull gemma3:1b
ollama pull schroneko/gemma-2-2b-jpn-it:q4_K_S
ollama pull deepseek-r1:8b
ollama pull qwen3:4b
```

For systems with CUDA-capable GPUs (premium hardware), also pull the large models:

```shell
ollama pull deepseek-r1:32b
ollama pull gemma3:27b
ollama pull qwen3:30b
ollama pull yuma/DeepSeek-R1-Distill-Qwen-Japanese:32b
```

## Installation

Install Python dependencies:

```shell
uv sync
```

Activate the virtual environment:

```shell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

## Embedding Vocabulary

Before generating content, populate the vector store with vocabulary words and sentences that comprise the learner's vocabulary inventory:

```shell
python -m src.scripts.embed_vocabulary
```

This embeds words, words with translations, and sentences into ChromaDB collections stored in `data/`. The script must be run once before using the content generation features.

The sentences are stored in [src/content_generation/generated_sentences_unique.txt](src/content_generation/generated_sentences_unique.txt). For copyright reasons, existing learning materials could not be used in this project. Instead, sentences were generated from a vocabulary word list using the Perplexity API.

## Running Timed Generation Scripts

The scripts benchmark text and question generation across different models and prompt configurations.

### Local Models (via Ollama)

Generate educational text:

```shell
python -m src.scripts.timed_text_generation_local
```

Generate comprehension questions:

```shell
python -m src.scripts.timed_questions_generation_local
```

### Perplexity API

Generate educational text:

```shell
python -m src.scripts.timed_text_generation_perplexity
```

Generate comprehension questions:

```shell
python -m src.scripts.timed_questions_generation_perplexity
```

Logs are written to the `logs/` directory with execution times.

## Web Server

To start the web server, run the following command in your terminal from the project's root directory:

```shell
uvicorn web:app --host 0.0.0.0 --reload --app-dir src
```

The `--host 0.0.0.0` option is necessary to provide access to the server via the local network. Check the IP of the computer on which the server is running in the network settings, e.g. `192.168.84.51`, and test the connection in the command line. On macOS or Linux, you can run the following command in the terminal:

```shell
curl -v http://192.168.84.51:8000/help
```

To verify that the server is accessible, send a POST request to the `/generate-test` endpoint to receive a canned response containing a text and a list of questions:

```shell
curl -sS X POST -H "Content-Type: application/json" -d '{"topic": "<specify_the_topic_here>"}' http://192.168.84.51:8000/generate-test
```

In PowerShell on Windows, the `text` or the `questions` property of the server response must be used to avoid the truncation of the response in the console:

```shell
(Invoke-RestMethod -Uri "http://192.168.84.58:8000/generate-test" -Method POST -ContentType "application/json" -Body '{"topic": "<specify_the_topic_here>"}').text
```

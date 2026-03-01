# Evaluation scripts

Scripts in this directory are used to evaluate the performance of local and cloud-based models across content creation tasks in JES:


Script |	Purpose
-------|----------------
timed_analysis_local.py | Benchmarks local LLM models for analysing user's answers (handwritten + spoken)
timed_analysis_perplexity.py |Benchmarks Perplexity API for user's answer analysis
timed_questions_generation_local.py | Benchmarks local LLMs for generating comprehension questions with long/short prompts
timed_questions_generation_perplexity.py | Benchmarks Perplexity API for question generation with long/short prompts
timed_text_generation_local.py | Benchmarks local LLMs for text generation with long/short prompts
timed_text_generation_perplexity.py | Benchmarks Perplexity API for text generation with long/short prompts

The embed_vocabulary.py script initialises vector databases by embedding Japanese vocabulary words and sentences into ChromaDB for semantic search. This script must be run once before starting the JES application to ensure the vocabulary database is populated and ready for use in the app's content generation functionality.

Read more about the workflow in each of the scripts.

"""
Embed vocabulary
================

Purpose:
    Initialises the vector database by embedding Japanese vocabulary words and sentences
    into ChromaDB collections for semantic similarity search. This is a one-time setup
    script that populates the retrieval database used by the application and must be perfomed
    before starting the JES application.

Workflow:
    1. Extracts Japanese words from the predefined vocabulary list.
    2. Embeds individual words into a ChromaDB collection for word-level lookups.
    3. Embeds word-translation pairs into a separate collection for bilingual search.
    4. Embeds pre-generated Japanese sentences for sentence-level retrieval.
    5. Tests each embedding by fetching similar entries to verify correctness.

Databases Created:
    - chroma_langchain_db_words: Individual Japanese words
    - chroma_langchain_db_words_with_translations: Word + English translation pairs
    - chroma_langchain_db_sentences: Japanese example sentences

Usage:
    Run this script once to initialize or rebuild the vector databases:
        python -m src.scripts.embed_vocabulary
    Remove existing ChromaDB collections if you want to re-embed with updated vocabulary or sentences.
    Failing to do so will result in duplicate entries in the database.
"""

from retrieval.embedding import embed_words, embed_words_with_translations, fetch_similar_entries, fetch_similar_words_with_translations
from content_generation.vocabulary import vocabulary_list
from content_generation.sentences import embed_clean_sentences

words = [word for word, _ in vocabulary_list]

print("Embedding words...")
embed_words(words)
print("Testing embedding by fetching similar entries for 'ちいさい':")
similar_entries = fetch_similar_entries("ちいさい", results_num=5)
print(similar_entries)

print("Embedding words with translations...")
embed_words_with_translations(vocabulary_list)
print("Testing embedding by fetching similar entries for ('大きい', 'big'):")
similar_entries_with_translations = fetch_similar_words_with_translations(("大きい", "big"), results_num=5)

print(similar_entries_with_translations)

print("Embedding clean sentences...")
embed_clean_sentences()
print("Testing embedding by fetching similar entries for '私は学生です。':")
similar_sentence_entries = fetch_similar_entries("私は学生です。", results_num=5, fetch_sentences=True)
print(similar_sentence_entries)

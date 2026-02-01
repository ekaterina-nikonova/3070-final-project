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

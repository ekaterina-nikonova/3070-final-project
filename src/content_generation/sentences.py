import json
from pathlib import Path

from dotenv import load_dotenv

from perplexity import Perplexity

from content_generation.missing_words import find_missing_words
from content_generation.vocabulary import vocabulary_dict
from retrieval.embedding import fetch_similar_entries, embed_sentences

load_dotenv()

CURRENT_MODULE_DIRPATH = Path(__file__).parent.resolve()
LOG_DIRPATH = CURRENT_MODULE_DIRPATH.parent.parent / "logs"

def generate():
    vocabulary_words = list(vocabulary_dict.keys())

    generated_sentences_filepath = CURRENT_MODULE_DIRPATH / "generated_sentences_unique.txt"
    generated_sentences_str = generated_sentences_filepath.read_text()

    entries = find_missing_words(generated_sentences_str, vocabulary_words)

    client = Perplexity()  # Automatically uses PERPLEXITY_API_KEY
    for i, entry in enumerate(entries):
        # From the vector store, fetch the 100 most similar words to the entry to use as vocabulary.
        vocabulary_words = fetch_similar_entries(entry, results_num=100)
        vocabulary_str = "\n".join(vocabulary_words)

        print(i, ":", entry)
        prompt = (
            "Here is a list of Japanese words that are currently in my vocabulary:\n\n"
            
            f"{vocabulary_str}\n\n"
            
            f"Using only the words in the vocabulary, write a few sentences in Japanese sentences at A1-level containing the word: {entry} \n\n"
            "Use a range of grammatical forms in present and past tense. Example sentences:\n\n"
    
            "姉が一人います。\n"
            "昨日姉は綺麗なドレスを買いました。\n"
            "新しいドレスを買いましょう。\n"
            "新しいドレスを買いましょうか。\n"
            "ドレスはかわいくておしゃれです。\n\n"
            
            "Include some longer, descriptive sentences containing linking words like とき, ので, けど, だけ, for example:\n\n"
    
            "小学生のとき、ピアノを習いました。\n"
            "近いので、公園に行きましょう。\n"
            "日本語はむずかしいけど、おもしろいです。\n"
            "お茶だけはおいしかったです。\n\n"
            
            "You can also include negations by using such forms as じゃない, じゃなかった, ありません, いません. Some examples:\n\n"
    
            "きれいなタオルはありませんでした。\n"
            "かれはしんせつじゃなかったです。\n"
            "お茶はあたたかくないです。\n"
            "行きたくないです。\n"
            "学校にいきたくなかったです。\n\n"
            
            "Make sure only the words that are in the list are used, this is very important!\n"
            "Your response must only contain the generated sentences, nothing more."
        )

        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="sonar",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentences": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                }
                            }
                        },
                        "required": ["sentences"]
                    }
                }
            }
        )


        try:
            sentences = json.loads(completion.choices[0].message.content)["sentences"]
        except KeyError:
            sentences = []
        except json.decoder.JSONDecodeError:
            with open(LOG_DIRPATH / "perplexity/perplexity.log", "a", encoding="utf-8") as log_f:
                log_f.write(f"Failed response for entry {entry}\n\n")
                log_f.write(f"{completion.choices[0].message.content}\n\n")
            continue
        else:
            with open(LOG_DIRPATH / "perplexity/perplexity.log", "a", encoding="utf-8") as log_f:
                log_f.write(f"Response for entry {entry}\n\n")
                log_f.write(f"{completion.choices[0].message.content}\n\n")

        with open(CURRENT_MODULE_DIRPATH / "generated_sentences.txt", "a", encoding="utf-8") as f:
            for sentence in sentences:
                f.write(f"{sentence}\n")
            f.write("\n")


def clean_generated_sentences():
    """Remove hallucinations and duplicated sentences."""

    # Split the file into groups of sentences separated by blank lines.
    # If the length of a group exceeds a threshold of 10,
    # only keep the first 10 sentences from that group.

    source_filepath = CURRENT_MODULE_DIRPATH / "generated_sentences.txt"
    with open(source_filepath, encoding="utf-8") as f:
        contents = f.read()

    cleaned_sentences = []
    sentence_groups = contents.split("\n\n")
    for group in sentence_groups:
        cleaned_sentences.extend(group.splitlines()[:10])

    unique_sentences = list(set([s.strip() for s in cleaned_sentences if s.strip()]))

    target_filepath = CURRENT_MODULE_DIRPATH / "generated_sentences_unique.txt"
    with open(target_filepath, "w", encoding="utf-8") as f:
        for sentence in unique_sentences:
            f.write(f"{sentence}\n")


def embed_clean_sentences():
    with open(CURRENT_MODULE_DIRPATH / "generated_sentences_unique.txt", encoding='utf-8') as f:
        sentences = f.read().splitlines()
    embed_sentences(sentences)


if __name__ == "__main__":
    generate()

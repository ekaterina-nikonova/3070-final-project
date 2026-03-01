"""
Functions in this module are used for generating sentences for each entry in the vocabulary.
Since most existing learning materials for A1-level Japanese learners are protected by copyright, 
the Perplexity API is used to generate new sentences, which are then cleaned to remove hallucinations and duplicates, 
before they can be embedded into the vector store.
"""

import json
from pathlib import Path

from dotenv import load_dotenv

from perplexity import Perplexity

from content_generation.edu_content_perplexity import DEFAULT_PERPLEXITY_MODEL
from content_generation.missing_words import find_missing_words
from content_generation.vocabulary import vocabulary_dict
from retrieval.embedding import fetch_similar_entries, embed_sentences

# Load environment variables from .env file (required for Perplexity API key)
load_dotenv()


# The path to the logs directory in the project's root relative to this module: ../../logs
# This allows storing logs in the same location regardless of where this module is called from.
CURRENT_MODULE_DIRPATH = Path(__file__).parent.resolve()
LOG_DIRPATH = CURRENT_MODULE_DIRPATH.parent.parent / "logs"

def generate():
    """Generate example sentences for vocabulary words missing from the sentence corpus.
    
    This function identifies vocabulary words that don't appear in the existing
    generated sentences file, then uses the Perplexity API to generate new
    example sentences containing those missing words.
    """
    # Get all vocabulary words as a list.
    vocabulary_words = list(vocabulary_dict.keys())

    # Load the existing "clean" generated sentences to check for missing vocabulary.
    generated_sentences_filepath = CURRENT_MODULE_DIRPATH / "generated_sentences_unique.txt"
    generated_sentences_str = generated_sentences_filepath.read_text()

    # Find vocabulary words that don't appear in any existing sentence.
    entries = find_missing_words(generated_sentences_str, vocabulary_words)

    # Initialise the Perplexity API client.
    client = Perplexity()  # Automatically uses PERPLEXITY_API_KEY for authentication
    
    # Process each missing vocabulary word.
    for i, entry in enumerate(entries):
        # From the vector store, fetch the 100 most similar words to the entry to use as vocabulary.
        # This provides context words that can be used alongside the target word.
        vocabulary_words = fetch_similar_entries(entry, results_num=100)
        # Join vocabulary words with newlines for readable display in the prompt.
        vocabulary_str = "\n".join(vocabulary_words)

        # Print a progress indicator showing the current entry number and word.
        print(i, ":", entry)
        
        # Construct a detailed prompt asking the model to generate sentences
        # containing the target word with only allowed vocabulary.
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

        # Send the request to Perplexity API with structured JSON response format.
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            # Use the same model as for the text and questions generation to maintain consistency in output style and quality.
            model=DEFAULT_PERPLEXITY_MODEL,
            # Request structured JSON response with an array of sentences
            # to ensure the API returns sentences in a predictable format.
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
            # Parse the JSON response and extract the "sentences" array.
            sentences = json.loads(completion.choices[0].message.content)["sentences"]
        except KeyError:
            # If "sentences" key is missing, set to empty list.
            sentences = []
        except json.decoder.JSONDecodeError:
            # If JSON parsing fails, log the raw response for debugging and skip to next entry.
            with open(LOG_DIRPATH / "perplexity/perplexity.log", "a", encoding="utf-8") as log_f:
                log_f.write(f"Failed response for entry {entry}\n\n")
                log_f.write(f"{completion.choices[0].message.content}\n\n")
            continue
        else:
            # On success, log the response for this vocabulary entry.
            with open(LOG_DIRPATH / "perplexity/perplexity.log", "a", encoding="utf-8") as log_f:
                log_f.write(f"Response for entry {entry}\n\n")
                log_f.write(f"{completion.choices[0].message.content}\n\n")

        # Append the generated sentences to the output file separated by a blank line.
        with open(CURRENT_MODULE_DIRPATH / "generated_sentences.txt", "a", encoding="utf-8") as f:
            for sentence in sentences:
                f.write(f"{sentence}\n")
            f.write("\n")


def clean_generated_sentences():
    """Remove hallucinations and duplicated sentences.
    
    This function processes the raw generated sentences file to:
    1. Limit each group of sentences to 10 (the experience shows outputs longer than that are likely to be hallucinations)
    2. Remove duplicate sentences
    3. Write the cleaned, unique sentences to a new file
    """

    # Read the source file containing all generated sentences.
    source_filepath = CURRENT_MODULE_DIRPATH / "generated_sentences.txt"
    with open(source_filepath, encoding="utf-8") as f:
        contents = f.read()

    # Initialize list to collect cleaned sentences.
    cleaned_sentences = []
    # Split content into groups (each group is separated by blank lines).
    sentence_groups = contents.split("\n\n")
    for group in sentence_groups:
        # For each group, take only the first 10 lines to trim potential hallucinations.
        # Even if all sentences in a long output are valid, limiting to 10 ensures a
        # balanced dataset for embedding and prevents over-representation of individual entries.
        cleaned_sentences.extend(group.splitlines()[:10])

    # Remove duplicates by converting to a set and strip whitespace.
    # Filter out empty strings that may result from blank lines.
    unique_sentences = list(set([s.strip() for s in cleaned_sentences if s.strip()]))

    # Write the unique sentences to the output file.
    target_filepath = CURRENT_MODULE_DIRPATH / "generated_sentences_unique.txt"
    with open(target_filepath, "w", encoding="utf-8") as f:
        for sentence in unique_sentences:
            f.write(f"{sentence}\n")


def embed_clean_sentences():
    """Embed the cleaned sentences into the vector store for semantic search.
    """
    # Read all unique sentences from the cleaned file.
    with open(CURRENT_MODULE_DIRPATH / "generated_sentences_unique.txt", encoding='utf-8') as f:
        sentences = f.read().splitlines()
    # Create embeddings for each sentence and store in the vector database.
    embed_sentences(sentences)


# Main entry point when run as a script from the command line.
if __name__ == "__main__":
    generate()

import os
from pathlib import Path


from typing import Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

# Generate vector embeddings locally; for cloud-based models, use HuggingFaceEndpointEmbeddings,
# which will require an API key that can be set in the .env file as HUGGINGFACEHUB_API_TOKEN.
from langchain_huggingface import HuggingFaceEmbeddings


# The name of the HuggingFace sentence-transformer model used for generating embeddings.
# This multilingual model supports 50+ languages and produces 768-dimensional vectors.
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# The vector store database paths are defined relative to this module's location in the project tree
# as ../../data/chroma_langchain_db_*. This allows for a consistent directory structure and easy access to the databases.
CURRENT_MODULE_DIRPATH = Path(__file__).parent.resolve()
DATA_DIRPATH = CURRENT_MODULE_DIRPATH.parent.parent / "data"

# Default path for the Chroma vector database storing word embeddings (without translations).
DEFAULT_WORDS_DB_DIRPATH = DATA_DIRPATH / "chroma_langchain_db_words"
# Default path for the Chroma vector database storing word embeddings with their translations as metadata.
DEFAULT_WORDS_WITH_TRANSLATIONS_DB_DIRPATH = DATA_DIRPATH / "chroma_langchain_db_words_with_translations"
# Default path for the Chroma vector database storing sentence embeddings.
DEFAULT_SENTENCES_DB_DIRPATH = DATA_DIRPATH / "chroma_langchain_db_sentences"

# Load environment variables from a .env file (required for HuggingFace cloud API).
load_dotenv()


def embed_words(
    words: list[str],
    db_dirpath: Optional[Path] = None,
):
    """Embed words and store them in a Chroma vector database located at db_dirpath.

    Args:
        words: The words to be embedded.
        db_dirpath: The directory path where the Chroma vector database will be stored.
    """
    # Use the default words database path if no custom path is provided.
    if db_dirpath is None:
        db_dirpath = DEFAULT_WORDS_DB_DIRPATH
    # Convert each word string into a LangChain Document object with the word as its page_content.
    docs = [Document(page_content=word) for word in words]
    # Delegate to the embed() function to generate embeddings and store them in the database.
    embed(docs, db_dirpath)


def embed_words_with_translations(
    words: list[tuple[str, str]],
    db_dirpath: Optional[Path] = None,
):
    """Embed words and store them in a Chroma vector database located at db_dirpath.

    Args:
        words: The words to be embedded as a list of (word, translation) tuples.
        db_dirpath: The directory path where the Chroma vector database will be stored.
    """
    # Use the default words-with-translations database path if no custom path is provided.
    if db_dirpath is None:
        db_dirpath = DEFAULT_WORDS_WITH_TRANSLATIONS_DB_DIRPATH
    # Convert each (word, translation) tuple into a Document with the word as page_content
    # and the translation stored in metadata for later retrieval.
    docs = [Document(page_content=word, metadata={"translation": translation}) for word, translation in words]
    # Delegate to the embed() function to generate embeddings and store them in the database.
    embed(docs, db_dirpath)


def embed_sentences(
    sentences: list[str],
    db_dirpath: Optional[Path] = None,
):
    """Embed sentences and store them in a Chroma vector database located at db_dirpath.

    Args:
        sentences: The sentences to be embedded.
        db_dirpath: The directory path where the Chroma vector database will be stored.
    """
    # Use the default sentences database path if no custom path is provided.
    if db_dirpath is None:
        db_dirpath = DEFAULT_SENTENCES_DB_DIRPATH
    # Convert each sentence string into a LangChain Document object with the sentence as its page_content.
    docs = [Document(page_content=sentence) for sentence in sentences]
    # Delegate to the embed() function to generate embeddings and store them in the database.
    embed(docs, db_dirpath)


def embed(
    entries: list[Document],
    db_dirpath: Path,
) -> Chroma:
    """Embed entries and store them in a Chroma vector database located at db_dirpath.

    Args:
        entries: The entries to be embedded.
        db_dirpath: The directory path where the Chroma vector database will be stored.

    Returns:
        The Chroma vector store containing the embedded entries.
    """
    # Initialize the HuggingFace embedding model that will convert text into numerical vectors.
    # The model runs locally, while trust_remote_code allows loading custom model implementations
    # (see more here: https://huggingface.co/docs/hub/en/models-uploading).
    embedding_function = HuggingFaceEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        model_kwargs = {"trust_remote_code": True},
    )

    # Create a Chroma vector store from the provided documents:
    return Chroma.from_documents(
        entries,                            # the Document objects to be embedded
        embedding=embedding_function,       # the function used to convert documents to vectors
        persist_directory=str(db_dirpath),  # where to save the database on disk for later retrieval
    )


def fetch_similar_words_with_translations(
    entry: tuple[str, str],
    results_num: int = 50,
    retrieve_by_translation: bool = True,
) -> list[tuple[str, str]]:
    """Fetch similar words with translations from a Chroma vector database.

    Args:
        entry: The (word, translation) tuple similar to which entries are to be retrieved.
        results_num: The number of similar entries to retrieve.
        retrieve_by_translation: Whether to retrieve similar entries based on the translation of the word.
                                 If False, retrieves similar entries based on the word itself.

    Returns:
        A list of (word, translation) tuples representing the most similar entries.
    """
    # Set the database path to the words-with-translations store.
    db_dirpath = DEFAULT_WORDS_WITH_TRANSLATIONS_DB_DIRPATH

    # Obtain a retriever configured to return the specified number of similar results.
    retriever = get_vector_store_retriever(db_dirpath, results_num)

    # Choose which part of the entry to use for similarity search:
    # entry[1] is the translation (e.g., English meaning), entry[0] is the original word.
    term = (
        entry[1]
        if retrieve_by_translation else
        entry[0]
    )
    # Execute the similarity search by embedding the term and finding nearest neighbours in the vector space.
    most_similar_results = retriever.invoke(term)

    # Extract and return the word and its translation from each result Document.
    return [(res.page_content, res.metadata["translation"]) for res in most_similar_results]


def fetch_similar_entries(
    entry: str,
    results_num: int = 50,
    fetch_sentences: bool = False,
) -> list[str]:
    """Fetch similar entries from a Chroma vector database.

    Args:
        entry: The entry similar to which entries are to be retrieved.
        results_num: The number of similar entries to retrieve.
        fetch_sentences: Whether to retrieve similar sentences instead of similar words.

    Returns:
        A list of strings representing the most similar entries.
    """
    # Select the appropriate database path based on whether we're searching for sentences or words.
    db_dirpath =  (
        DEFAULT_SENTENCES_DB_DIRPATH
        if fetch_sentences else
        DEFAULT_WORDS_DB_DIRPATH
    )

    # Obtain a retriever configured to return the specified number of similar results.
    retriever = get_vector_store_retriever(db_dirpath, results_num)

    # Execute the similarity search by embedding the entry and finding nearest neighbours in the vector space.
    most_similar_results = retriever.invoke(entry)

    # Extract and return just the text content from each result Document.
    return [res.page_content for res in most_similar_results]


def get_vector_store_retriever(db_dirpath: Path, results_num: int) -> VectorStoreRetriever:
    """Get a retriever for a Chroma vector database.

    Args:
        db_dirpath: The directory path where the Chroma vector database is stored.
        results_num: The number of similar entries to retrieve.

    Returns:
        A retriever for the Chroma vector database.
    """
    # Verify that the database directory exists before attempting to load it.
    if not os.path.exists(db_dirpath):
        raise ValueError(f"Chroma database at {db_dirpath} does not exist. You must embed entries first.")

    # Initialise the same embedding model used during database creation to ensure vector compatibility.
    # IMPORTANT! Query embeddings must use the same model as the stored document embeddings 
    # for accurate similarity matching.
    embedding_function = HuggingFaceEmbeddings(
        model=EMBEDDING_MODEL_NAME,
    )

    # Load the existing Chroma vector store from disk using the persist_directory.
    # The embedding_function is needed to convert query strings into vectors during retrieval.
    vector_store = Chroma(
        persist_directory=str(db_dirpath),
        embedding_function=embedding_function,
    )

    # Create a retriever interface with similarity-based search.
    retriever = vector_store.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": results_num},  # how many nearest neighbours to return for each query
    )
    return retriever

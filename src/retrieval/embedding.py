import os
from pathlib import Path


from typing import Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEndpointEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

CURRENT_MODULE_DIRPATH = Path(__file__).parent.resolve()
DATA_DIRPATH = CURRENT_MODULE_DIRPATH.parent.parent / "data"
DEFAULT_WORDS_DB_DIRPATH = DATA_DIRPATH / "chroma_langchain_db_words"
DEFAULT_WORDS_WITH_TRANSLATIONS_DB_DIRPATH = DATA_DIRPATH / "chroma_langchain_db_words_with_translations"
DEFAULT_SENTENCES_DB_DIRPATH = DATA_DIRPATH / "chroma_langchain_db_sentences"

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
    if db_dirpath is None:
        db_dirpath = DEFAULT_WORDS_DB_DIRPATH
    docs = [Document(page_content=word) for word in words]
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
    if db_dirpath is None:
        db_dirpath = DEFAULT_WORDS_WITH_TRANSLATIONS_DB_DIRPATH
    docs = [Document(page_content=word, metadata={"translation": translation}) for word, translation in words]
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
    if db_dirpath is None:
        db_dirpath = DEFAULT_SENTENCES_DB_DIRPATH
    docs = [Document(page_content=sentence) for sentence in sentences]
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
    embedding_function = HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL_NAME,

    )

    return Chroma.from_documents(
        entries,
        embedding=embedding_function,
        persist_directory=str(db_dirpath),
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
    db_dirpath = DEFAULT_WORDS_WITH_TRANSLATIONS_DB_DIRPATH

    retriever = get_vector_store_retriever(db_dirpath, results_num)

    term = (
        entry[1]
        if retrieve_by_translation else
        entry[0]
    )
    most_similar_results = retriever.invoke(term)

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
    db_dirpath =  (
        DEFAULT_SENTENCES_DB_DIRPATH
        if fetch_sentences else
        DEFAULT_WORDS_DB_DIRPATH
    )

    retriever = get_vector_store_retriever(db_dirpath, results_num)

    most_similar_results = retriever.invoke(entry)

    return [res.page_content for res in most_similar_results]


def get_vector_store_retriever(db_dirpath: Path, results_num: int) -> VectorStoreRetriever:
    """Get a retriever for a Chroma vector database.

    Args:
        db_dirpath: The directory path where the Chroma vector database is stored.
        results_num: The number of similar entries to retrieve.

    Returns:
        A retriever for the Chroma vector database.
    """
    if not os.path.exists(db_dirpath):
        raise ValueError(f"Chroma database at {db_dirpath} does not exist. You must embed entries first.")

    embedding_function = HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL_NAME,
    )

    vector_store = Chroma(
        persist_directory=str(db_dirpath),
        embedding_function=embedding_function,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": results_num}
    )
    return retriever


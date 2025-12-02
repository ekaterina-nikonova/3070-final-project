import os
from pathlib import Path


from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEFAULT_WORDS_DB_DIRPATH = Path("./chroma_langchain_db_words")
DEFAULT_SENTENCES_DB_DIRPATH = Path("./chroma_langchain_db_sentences")


def embed_words(
    words: list[tuple[str, str]],
    db_dirpath: Optional[Path] = None,
):
    """Embed words and store them in a Chroma vector database located at db_dirpath.

    Args:
        words: The words to be embedded as (word, translation) tuples.
        db_dirpath: The directory path where the Chroma vector database will be stored.
    """
    if db_dirpath is None:
        db_dirpath = DEFAULT_WORDS_DB_DIRPATH
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


def fetch_similar(
    entry: str | tuple[str, str],
    results_num: int = 50,
) -> list[str]:
    db_dirpath = (
        DEFAULT_SENTENCES_DB_DIRPATH
        if isinstance(entry, str)
        else DEFAULT_WORDS_DB_DIRPATH
    )

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

    most_similar_results = retriever.invoke(entry)

    return [res.page_content for res in most_similar_results]

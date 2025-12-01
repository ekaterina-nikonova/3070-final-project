def find_missing_words(sentences: str, vocabulary: list[str]) -> list[str]:
    missing = []
    for entry in vocabulary:
        if not entry in sentences:
            missing.append(entry)

    return missing

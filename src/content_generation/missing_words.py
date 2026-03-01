def find_missing_words(sentences: str, vocabulary: list[str]) -> list[str]:
    """Find vocabulary words that are not present in the given sentences.
    
    Args:
        sentences: A string containing all sentences to search within.
        vocabulary: A list of vocabulary words/phrases to look for.
    
    Returns:
        A list of vocabulary entries that were not found in the sentences.
    """
    # Hold the collected words that are missing from the sentences.
    missing = []
    # In each vocabulary entry...
    for entry in vocabulary:
        # ...check if the entry is NOT found anywhere in the sentences string using substring matching.
        if not entry in sentences:
            # Add the missing entry to the list.
            missing.append(entry)

    # Return the list of all vocabulary entries not found in the sentences.
    return missing

from content_generation.vocabulary import vocabulary_dict
from retrieval.embedding import fetch_similar_entries


DEFAULT_TEXT_LENGTH_CHAR = 200


# The short prompt functions are designed for smaller models that are invoked locally, 
# and they provide more concise instructions and fewer examples, as well as 
# a shorter list of vocabulary sentences.

def make_text_system_message_short(topic: str) -> str:
    """A simple system message suitable for small models that are invoked locally."""
    vocabulary_words = fetch_similar_entries(topic, results_num=10, fetch_sentences=True)
    vocabulary_str = "\n".join(vocabulary_words)

    return (
        "You are a Japanese teacher creating reading material for JLPT N5 (beginner) students.\n"
        "The user will provide a topic, and you have to write a short story on that topic.\n\n"
        
        "Instructions:\n\n"
        "- Write a story between 100 and 200 characters long.\n"
        "- Use simple grammar.\n"
        "- Try to include these specific phrases naturally:\n\n"
        
        f"{vocabulary_str}\n\n"
        
        "### EXAMPLE:\n\n"
        "User:\n"
        "Topic: 週末\n\n"
        "Assistant:\n"
        "私は土曜日に映画を見ました。アクション映画でした。とても楽しかったです。日曜日は家で本を読みました。いい週末でした。\n"
    )


def make_text_user_message_short(topic: str) -> str:
    """A simple user message suitable for small models that are invoked locally."""
    return (
        f"Write a story about this topic: {topic}\n\n"
        "Remember to use simple grammar and keep it under 200 characters."
    )


def make_questions_system_message_short() -> str:
    """A simple system message suitable for small models that are invoked locally."""
    return (
        "You are a Japanese teacher.\n\n"
        
        "Task: Read the text provided by the user and write 3 simple comprehension questions in Japanese.\n\n"
        
        "Constraints:\n"
        "- The questions must be easy to answer using the text provided.\n"
        "- Use the same vocabulary found in the text.\n"
        "- Do not use complex Kanji.\n"
        
        "Examples:\n\n"
        "User Text: 私は昨日、公園へ行きました。\n"
        "Question: 昨日はどこへ行きましたか？\n\n"
        
        "User Text: 朝ごはんはパンを食べました。\n"
        "Question: 朝ごはんに何を食べましたか？\n\n"
        
        "### GUIDELINES:\n\n"
        "1. Read the text carefully.\n"
        "2. Create 3 questions in Japanese that can be answered directly from the text.\n"
        "3. Use simple vocabulary (A1/N5 level).\n"
        "4. Do NOT provide the answers.\n"
        "5. Output ONLY a numbered list."
    )


def make_questions_user_message_short(text: str) -> str:
    """A simple user message suitable for small models that are invoked locally."""
    return (
        "Here is a text for a student:\n\n"
        f"{text}\n\n"
        "Please generate 3 questions based on the text above."
    )


# The following functions are for the long prompt, which is used for large models and the Perplexity API. 
# They contain more detailed instructions and examples, and they also include a longer list 
# of vocabulary sentences to use in the generated text.

# In case of long prompts, the user message for the text generation is simply the topic, 
# since the system message already contains detailed instructions and examples.
# For the same reason, the user message for the question generation will only contain the text 
# for which questions are to be generated.


def make_text_system_message(topic: str) -> str:
    # From the vector store, fetch 50 sentences related to the topic to use as vocabulary.
    vocabulary_words = fetch_similar_entries(topic, results_num=50, fetch_sentences=True)
    vocabulary_str = "\n".join(vocabulary_words)

    return (
        "You are a creator of educational Japanese content for A1-level learners. "
        "Your task is to write a short, cohesive text in Japanese on the topic the user provides. "
        "The text must be simple, clear, and appropriate for beginners.\n\n"
        "Only use words and phrases from the following sentences:\n\n"

        f"{vocabulary_str}\n\n"

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

        f"The text must contain up to {DEFAULT_TEXT_LENGTH_CHAR} characters, be cohesive and tell an engaging story."

        "Your response must only contain the text you write, nothing more."
    )


def make_questions_system_message() -> str:
    vocabulary = "\n".join([f"{word} ({translation})" for word, translation in vocabulary_dict.items()])

    return (
        "You are a creator of educational Japanese content for A1-level learners. "
        "Your task is to write three comprehension questions for the text that the user provides.\n\n"
        "Only use words and phrases from the following vocabulary:\n\n"

        f"{vocabulary}\n\n"

        "The questions must check the learner's understanding of the text, be concise and appropriate for beginners. "
        "Your response must contain only a list of questions, nothing more."
    )

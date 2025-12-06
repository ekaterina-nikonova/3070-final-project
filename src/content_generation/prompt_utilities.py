from content_generation.vocabulary import vocabulary_dict
from retrieval.embedding import fetch_similar_entries


def make_text_system_message(topic: str) -> str:
    # From the vector store, fetch 20 sentences related to the topic to use as vocabulary.
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

        "The text must contain from 200 to 500 words, be cohesive and tell an engaging story."

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

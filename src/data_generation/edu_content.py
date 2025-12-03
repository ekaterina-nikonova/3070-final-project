import json

from dotenv import load_dotenv
from huggingface_hub.errors import GenerationError
from perplexity import Perplexity

from retrieval.embedding import fetch_similar_entries

load_dotenv()

def generate_text(topic: str):
    if not topic:
        raise ValueError("Topic must be specified.")

    client = Perplexity()  # Automatically uses PERPLEXITY_API_KEY

    # From the vector store, fetch 20 sentences related to the topic to use as vocabulary.
    vocabulary_words = fetch_similar_entries(topic, results_num=50, fetch_sentences=True)
    vocabulary_str = "\n".join(vocabulary_words)

    print("Vocabulary sentences:\n", vocabulary_str)

    prompt = (
        f"Write an A1-level text in Japanese on the topic {topic!r}.\n\n"
        
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
                        "text": {
                            "type": "string",
                        }
                    },
                    "required": ["text"]
                }
            }
        }
    )

    try:
        text = json.loads(completion.choices[0].message.content)["text"]
    except KeyError:
        raise GenerationError("Failed to generate a text.")
    except json.decoder.JSONDecodeError:
        with open("perplexity.log", "a") as log_f:
            log_f.write(f"Failed to generate a text for topic: {topic!r}\n")
            log_f.write(f"{completion.choices[0].message.content}\n\n")
    else:
        print(text)
        return text
import json
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from pathlib import Path
from uuid import uuid4
from pydantic import BaseModel

from content_generation.edu_content_perplexity import generate_text, generate_questions
from assessment.analysis import analyse_answers

class GenerateRequest(BaseModel):
    topic: str

class GenerateResponse(BaseModel):
    text: str
    questions: List[str]

app = FastAPI()

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required")

    # Generate text and questions using existing functions
    text = generate_text(topic)
    questions = generate_questions(text)

    return GenerateResponse(text=text, questions=questions)


@app.post("/generate-test", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required")

    # Use a canned response for testing
    test_response = json.loads("""
    {"text": "今日は母の誕生日です。誕生日、おめでとうございます。お母さんは四十五歳ぐらいの歳です。家族といるとしあわせです。母が来ました。ただいま。昨日母に電話しました。お母さんから手紙が来ました。けいたいでしゃしんをとります。あたらし いけいたいはおおきいです。けいたいででんわをします。家族と一緒にいます。来週、母の誕生日パーティーがあります。こうがいでさんぽします。こうがいはとおいけど、たのしいです。お母さんが作ったとろとろのオムライスを食べました。おべんとうを一 つください。おべんとう、あたためましょうか。おさけはあまいです。くすりはにがいです。お母さんは何もしませんでした。今日はさいあくの日でした。今日はダメじゃないです。せいせきがいいので、うれしいです。小さいとき、お母さんはしんせつでした 。あたらしいゆびわをあげます。お母さんはうれしそうです。姉はえいごがとくいです。姉が一人います。先週の誕生日に、姉がおごりでケーキを買いました。姉はえをかくのが上手です。えをかくのはたのしいです。七日は私の誕生日です。二十日は友達の誕 生日です。九日は友達の誕生日です。昨日は友達の誕生日でした。お母さんはおいくつですか。四十五歳です。おおみそかは家族と一緒にいます。かいしゃにいくまえに、でんきをあけます。すわります。おこりました。今日は手がベタベタです。じこはきらい です。かわいいみみがあります。うさぎを愛します。しんぷはしんせつなので、みんながすきです。しんぷはしんせつです。あたまがいいです。やっつ の子供がいます。あかちゃんのとき、よくねました。きれいなけっこんしきでした。しょうしょうおまちください。ゆびわをあげます。今日、よやくを入れます。こわれます。へんにちは休みです。お母さんはいつも優しいです。家族みんなでお祝いします。お母さん、ありがとうございます。", "questions": ["今日は**お母さん**の**誕生日**ですか。", "お母さん は**何歳**ぐらいですか。", "**姉**は何が**とくい**ですか。", "**七日**は**誰**の**誕生日**ですか。"]}
    """)
    text = test_response["text"]
    questions = test_response["questions"]

    return GenerateResponse(text=text, questions=questions)



UPLOAD_DIR = Path(__file__).parent / "uploads" / "answers"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/submit-answer")
async def submit_answer(
    text: str = Form(...),
    selected_question: str = Form(...),
    handwritten: UploadFile = File(...),
    spoken: UploadFile = File(...),
):
    # Basic validation of required form fields
    if not text.strip():
        raise HTTPException(status_code=400, detail="`text` is required")
    if not selected_question.strip():
        raise HTTPException(status_code=400, detail="`selected_question` is required")

    # Validate content types/extensions
    hw_suffix = Path(handwritten.filename).suffix.lower()
    sp_suffix = Path(spoken.filename).suffix.lower()
    if hw_suffix != ".png" and not (handwritten.content_type and "image" in handwritten.content_type):
        raise HTTPException(status_code=400, detail="Handwritten file must be a PNG image")
    if sp_suffix not in (".wav",) and not (spoken.content_type and "wav" in spoken.content_type):
        raise HTTPException(status_code=400, detail="Spoken file must be a WAV audio")

    # Save files with unique names
    hw_path = UPLOAD_DIR / f"{uuid4().hex}{hw_suffix or '.png'}"
    sp_path = UPLOAD_DIR / f"{uuid4().hex}{sp_suffix or '.wav'}"

    try:
        hw_bytes = await handwritten.read()
        sp_bytes = await spoken.read()

        hw_path.write_bytes(hw_bytes)
        sp_path.write_bytes(sp_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded files: {e}")

    # Call analyse_answers in threadpool (analysis is synchronous / may block)
    try:
        # feedback = await run_in_threadpool(
        #     analyse_answers, text, selected_question, str(hw_path), str(sp_path)
        # )
        feedback = "All good! ありがとうございます。"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    return Response(content=feedback, media_type="text/plain; charset=utf-8")

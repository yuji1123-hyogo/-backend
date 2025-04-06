from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, List, TypedDict
from src import gemini

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class History(TypedDict):
    role: Literal["user", "model"]
    parts: List[str]


class QuestionItem(BaseModel):
    history: List[History]


@app.post("/api/question")
async def question(item: QuestionItem):
    history = item.history

    prompt = f"""これまでの内容やユーザーの思考のジャンル・対象・ゴールなどをもとに、思考をより深めるための質問をしてください。"""

    return await request(history, prompt)


class ReactionItem(BaseModel):
    history: List[History]
    question: str
    answer: str


@app.post("/api/reaction")
async def reaction(item: ReactionItem):
    history = item.history
    question = item.question
    answer = item.answer

    prompt = f"""{question}

        というあなたの質問に対して、以下のような回答がありました。

        {answer}

        "この回答に対しての返答し、次の質問を投げてください。"
    """

    return await request(history, prompt)


class evaluateItem(BaseModel):
    history: List[History]
    sentence: str
    evaluate_type: List[str]


@app.post("/api/evaluate")
async def evaluate(item: evaluateItem):
    history = item.history
    sentence = item.sentence
    evaluate_type = item.evaluate_type

    prompt = f"""\
            これまでの内容を踏まえて、下記の文章を作成しました。この文章を評価してください。

            ${sentence}

            ただし、評価項目は以下の通りです。

            ${evaluate_type}
        """

    return await request(history, prompt)


async def request(history: List[History], prompt: str):
    request_dict = history + [{"role": "user", "parts": [prompt]}]
    response = gemini.generate(request_dict)

    new_history = history + [
        {"role": "user", "parts": [prompt]},
        {"role": "model", "parts": [response]},
    ]

    return {
        "prompt": prompt,
        "response": response,
        "history": new_history,
    }

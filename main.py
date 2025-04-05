from fastapi import FastAPI
from pydantic import BaseModel
from src import gemini

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


class QuestionItem(BaseModel):
    history: list[str]


@app.post("/api/question")
def question(item: QuestionItem):
    print(item)
    history = item.history

    prompt = f"""\
        これまでの内容やユーザーの思考のジャンル・対象・ゴールなどをもとに、思考をより深めるための質問をしてください。
    """

    return request(history, prompt)


class ReactionItem(BaseModel):
    history: list[str]
    question: str
    answer: str


@app.post("/api/reaction")
async def reaction(item: ReactionItem):
    history = item.history
    question = item.question
    answer = item.answer

    prompt = f"""\
        {question}

        という質問に対して、以下のような回答がありました。

        {answer}

        この回答に対しての返答をしてください。
    """

    return request(history, prompt)


class evaluateItem(BaseModel):
    history: list[str]
    sentence: str
    evaluate_type: list[str]


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

    return request(history, prompt)


async def request(history, prompt):
    request_dict = history + {"role": "user", "parts": [prompt]}
    response = gemini.gemini(request_dict).text

    new_history = history + [
        {"role": "user", "parts": [prompt]},
        {"role": "model", "parts": [response]},
    ]

    return {
        "prompt": prompt,
        "response": response,
        "history": new_history,
    }


# 動作チェック用のコード
def check():
    history = [
        {
            "role": "system",
            "parts": [
                "あなたは、ユーザーの思考の整理・言語化・ブラッシュアップのサポートをするAIです。\nユーザーがより良い体験ができることを最優先に考えて生成してください。\n対話はすべて日本語で行ってください。"
            ],
        },
        {
            "role": "user",
            "parts": [
                "私は以下のことを言語化したいと考えています。\n\n- 言語化したい内容：企業のエントリーシートに記述する自己PR\n- 言語化のゴール：エントリーシートに書けるような文章"
            ],
        },
    ]
    question(history)


if __name__ == "__main__":
    print(check())

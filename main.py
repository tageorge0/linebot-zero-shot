# LINE Bot × Zero-shot Text Classification
# Flask + Hugging Face pipeline + log 記錄

from flask import Flask, request, abort

from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent
)

import csv
import os
from datetime import datetime
import requests

app = Flask(__name__)
configuration = Configuration(access_token=os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler =  WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))
# Hugging Face API 設定
HF_API_URL = "https://router.huggingface.co/hf-inference/models/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
HF_HEADERS = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}

# 紀錄結果到 CSV
LOG_FILE = "emotion_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["時間", "使用者ID", "輸入文字", "分類結果", "信心分數"])

def log_emotion(user_id, text, label, score):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), user_id, text, label, round(score, 4)])

# 呼叫 Hugging Face Zero-shot API
def classify_text(text):
    labels = ["正面", "負面"]
    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": labels
        }
    }
    try:
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=5)
        result = response.json()
        label = result["labels"][0]
        score = result["scores"][0]
        return label, score
    except Exception as e:
        print("Hugging Face API 發生錯誤：", e)
        return "無法判斷", 0.0

# Webhook
@app.route("/")
def hello():
	return "Hello World"

# Webhook callback
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.info("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_id = event.source.user_id
    user_text = event.message.text.strip()

    if not user_text:
        summary = "請輸入一句話來分析情感喔～"
    else:
        label, score = classify_text(user_text)
        summary = f"這句話是「{label}」情感（信心：{round(score * 100)}%)"
        log_emotion(user_id, user_text, label, score)

    with ApiClient(configuration) as api_client:
        messaging_api = MessagingApi(api_client)
        try:
            messaging_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=summary)]
                )
            )
        except Exception as e:
            print("回覆訊息失敗：", e)

if __name__ == '__main__':
	    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
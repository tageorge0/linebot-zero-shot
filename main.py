# LINE Bot × Zero-shot Text Classification
# Flask + Hugging Face pipeline + log 記錄

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import csv
import os
from datetime import datetime
from dotenv import load_dotenv
import requests
from threading import Thread



line_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))
# Hugging Face API 設定
HF_API_URL = "https://api-inference.huggingface.co/models/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
HF_HEADERS = {"Authorization": f"Bearer {os.getenv("HF_API_TOKEN")}"}

app = Flask(__name__)

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
            "candidate_labels": labels,
            "hypothesis_template": "這句話的情感是 {}。"
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

# Webhook callback
@app.route("/callback", methods=['POST'])
def callback():
    print("收到 LINE Webhook 請求")
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    Thread(target=safe_handle, args=(body, signature)).start()
    return 'OK', 200

# 將 handler.handle 包裝起來
def safe_handle(body, signature):
    try:
        handler.handle(body, signature)
        print("handler 處理完成")
    except Exception as e:
        print("handler 處理錯誤：", e)

# 處理文字訊息（非同步）
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    print(f"收到使用者輸入：{event.message.text}")
    user_input = event.message.text
    user_id = event.source.user_id

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="訊息收到！分析中..."))

    # 背景處理
    def background_analysis():
        try:
            label, score = classify_text(user_input)
        except Exception as e:
            print("分類失敗：", e)
            label, score = "無法判斷", 0.0
        log_emotion(user_id, user_input, label, score)

        # 回傳分析結果給使用者（使用 push_message）
        summary = f"這句話是「{label}」情感（信心：{round(score, 2)}）"
        line_bot_api.push_message(user_id, TextSendMessage(text=summary))

    Thread(target=background_analysis).start()

if __name__ == '__main__':
	    app.run(debug=True, port=os.getenv("PORT", default=5000))
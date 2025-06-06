# LINE Bot × Sentiment Classification (local model)
# -------------------------------------------------------------------
# 1. 啟動時自動下載 Hugging Face 上的私有/公開模型
# 2. 本機用 transformers 直接推論，不走外部 API
# 3. 結果寫入 CSV、回傳給使用者
# -------------------------------------------------------------------

import os, csv
from datetime import datetime

from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi,
    ReplyMessageRequest, TextMessage
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent

# -------------------- (A) 下載並載入模型 ----------------------------
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, torch.nn.functional as F

# ① 你的模型 repo ID
MODEL_REPO = "TAGeorge/my-sentiment-model"

# ② 下載到 Railway Volume（或臨時目錄亦可）
MODEL_DIR = snapshot_download(repo_id=MODEL_REPO)   # 公開 repo → 不需要 token

# ③ 載入
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# ④ id ↔ label 映射（看你的 config.json）
id2label = {0: "negative", 1: "neutral", 2: "positive"}

# -------------------- (B) Flask + LINE Bot --------------------------
app = Flask(__name__)
configuration = Configuration(access_token=os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))

# -- CSV log
LOG_FILE = "emotion_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["時間", "使用者ID", "輸入文字", "分類結果", "信心分數"])

def log_emotion(user_id, text, label, score):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.now(), user_id, text, label, round(score, 4)])

# -- 本機推論函式
def classify_text_local(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        probs = F.softmax(model(**inputs).logits, dim=1)
    cls_id = int(torch.argmax(probs))
    label = id2label[cls_id]
    score = float(probs[0][cls_id])
    return label, score

# -- Health check
@app.route("/")
def hello():
    return "Hello World"

# -- LINE Webhook
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_id = event.source.user_id
    user_text = event.message.text.strip()

    if not user_text:
        summary = "請輸入一句話來分析情感喔～"
    else:
        label, score = classify_text_local(user_text)
        summary = f"這句話是「{label}」情感（信心：{round(score*100)} %）"
        log_emotion(user_id, user_text, label, score)

    with ApiClient(configuration) as api_client:
        MessagingApi(api_client).reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=summary)]
            )
        )

if __name__ == "__main__":
    # Railway 預設會把 PORT 寫入環境變數
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=False)

# 第一階段：用來安裝 Python 套件
FROM python:3.11-slim AS builder
WORKDIR /app

# 安裝必要系統工具
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 安裝 Python 套件（不保留快取）
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 複製你的程式進來
COPY . .

# 第二階段：正式執行用
FROM python:3.11-slim
WORKDIR /app

# 拷貝 Python 環境與程式
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

# 啟動應用
CMD ["python", "main.py"]

FROM python:3.9-slim

WORKDIR /app

# 創建 ChromaDB 資料夾
RUN mkdir -p /data/chromadb

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"] 
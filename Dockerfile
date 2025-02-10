FROM python:3.9-slim

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y supervisor

# 複製專案文件
COPY requirements.txt .
COPY app.py .
COPY vanna-web.py .
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install supervisor

# 確保日誌目錄存在
RUN mkdir -p /var/log/supervisor

# 使用 supervisor 啟動服務
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"] 
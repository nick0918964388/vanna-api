from fastapi import FastAPI
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore
import psycopg2
import pandas as pd
from typing import Dict

# 建立 FastAPI 應用
app = FastAPI(title="Vanna API", description="AI 資料庫查詢 API")

class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

# 初始化 MyVanna
vn = MyVanna(config={
    'model': 'llama3.2-vision',
    'ollama_host': "http://ollama.webtw.xyz:11434"
})

# 連接到 SQLite 資料庫
#vn.connect_to_postgres(host='10.10.10.168', dbname='postgres', user='admin', password='admin', port='5432')

def run_sql(sql: str) -> pd.DataFrame:
    with psycopg2.connect(
        host='10.10.10.168',
        database='postgres',
        user='admin',
        password='admin',
        port='5432'
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            result = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(result, columns=columns)
            return df

vn.run_sql = run_sql
vn.run_sql_is_set = True

# API 端點
@app.get("/")
async def root():
    return {"message": "歡迎使用 Vanna API"}

@app.post("/ask")
async def ask_question(question: Dict[str, str]):
    try:
        response = vn.ask(question["query"])
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

from fastapi import FastAPI, Body
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore
import psycopg2
import pandas as pd
from typing import Dict, Optional

# 建立 FastAPI 應用
app = FastAPI(title="Vanna API", description="AI 資料庫查詢 API")

class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

# 初始化 MyVanna
vn = MyVanna(config={
    'model': 'mistral-small:latest',
    'ollama_host': "http://ollama.webtw.xyz:11434",
    'path': '/data/chromadb'
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
        # response = vn.ask(question["query"])
        res = vn.ask(
            question["query"], print_results=True, auto_train=True, visualize=False, allow_llm_to_see_data=False
        )
        # response = []
        # if res is not None:
        #     response.append(res[0])
        #     if len(res) > 1 and res[1] is not None:
        #         response.append(res[1])
        print('res',res)
        # 將結果轉換為純文字格式
        if isinstance(res, (list, tuple)):
            text_response = []
            for item in res:
                if isinstance(item, pd.DataFrame):
                    text_response.append(item.to_string())
                else:
                    text_response.append(str(item))
            res = "\n".join(text_response)
        else:
            res = str(res)
        return res
    except Exception as e:
        return {"error": str(e)}


@app.post("/train")
async def train_model(
    ddl: Optional[str] = Body(None),
    sql: Optional[str] = Body(None),
    question: Optional[str] = Body(None),
    memos: Optional[str] = Body(None)
):
    try:
        response = {}
        
        if ddl:
            vn.train(ddl=ddl)
            response["ddl"] = "DDL 訓練完成"

        if sql:
            if question:
                vn.train(question=question, sql=sql)
                response["sql_with_question"] = "SQL 和問題配對訓練完成"
            else:
                vn.train(sql=sql)
                response["sql"] = "SQL 訓練完成"
                
        if memos:
            vn.train(documentation=memos)
            response["memos"] = "文件訓練完成"
            
        if not response:
            return {"message": "沒有提供訓練資料"}
            
        return {"message": "訓練成功", "details": response}
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

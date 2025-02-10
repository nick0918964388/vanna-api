from vanna.flask import VannaFlaskApp
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore
import psycopg2
import pandas as pd
from typing import Dict, Optional

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

app = VannaFlaskApp(vn)
app.run()
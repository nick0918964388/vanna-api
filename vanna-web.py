from vanna.flask import VannaFlaskApp
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore

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

app = VannaFlaskApp(vn)
app.run()
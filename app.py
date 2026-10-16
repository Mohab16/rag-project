from src.data_loader import load_documents
from src.vector_store import FaissVectorStore
from src.search import RAGSearch
from fastapi import FastAPI
from api.schemas import QueryResposnse

    
app = FastAPI()
rag_search = RAGSearch()
store = FaissVectorStore("faiss_store")
@app.get("/")
def home():
    return {"message": "Customized Chatbot is running"}
@app.get("/index")
def index():
    docs = load_documents("data")
    store.build_from_documents(docs)
    store.save()
    store.load()
    return {"message": "Indexing is done"}

@app.post("/query", response_model=QueryResposnse)

async def answer(query: str):
    
    #print(store.query("What is attention mechanism?", top_k=3))
    summary = rag_search.search_and_summarize(query, top_k=3)
    
    return QueryResposnse(
        response=summary
    )
        
        
    


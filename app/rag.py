from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import psycopg2

#* Database connection parameters
DBNAME = 'postgres'
USER = 'devadmin'
PASSWORD = 'KCE9MP2En93gLCz2'
HOST = 'bhyve-india-dev-db.postgres.database.azure.com'
PORT = '5432'


# Models 
embedding_model = 'bert-base-nli-mean-tokens'
llm_model = './models/llama-2-7b.Q4_0.gguf'
embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

# Callback Streaming Function
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


# Load Models
llm = LlamaCpp(model_path=llm_model, n_gpu_layers=43, n_batch=1024, n_ctx=4096, callback_manager=callback_manager, verbose=False)
model = SentenceTransformer(embedding_model)
embed_model = HuggingFaceEmbeddings(model_name=embed_model_id, encode_kwargs={'batch_size': 32})

prompt_template_qa = """
  Use the following pieces of context to answer the question at the end. Make sure you form complete sentences. Do not given broken sentences. If you don't know the answer, just say that you don't know, don't try to make up an answer.

  {context}

  Question: {question}
  Answer:

"""


# Fetch Splits for Query
def get_search_results(query):
  #* Connect to the database
  connection = psycopg2.connect(dbname=DBNAME, user=USER, password=PASSWORD, host=HOST, port=PORT)
  conn = connection.cursor()
  query_embeddings = model.encode([query.lower()])[0]
  conn.execute(f"SELECT content_id, content FROM pg_embed ORDER BY embedding <=> '{query_embeddings.tolist()}' LIMIT 5")
  results = conn.fetchall()
  conn.close()
  connection.close()
  return results

def search_results_as_doc_arr(query):
  results = get_search_results(query)
  splits = []
  for result in results:
    doc =  Document(page_content=result[1], metadata={"source": result[0]})
    splits.append(doc)
  print('Splits: ', len(splits))
  return splits


def answer(query):
  print('Query: ', query)
  splits = search_results_as_doc_arr(query)
  vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model)
  PROMPT = PromptTemplate(template=prompt_template_qa, input_variables=["context", "question"])
  chain_type_kwargs = {"prompt": PROMPT}
  qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(), chain_type_kwargs=chain_type_kwargs)
  print('Processing')
  result = qa(query)
  return result

app = FastAPI()

class SearchPayload(BaseModel):
  query: str

@app.get("/")
def Home():
    return {"Hello": "World"}

@app.post("/answer")
def search(payload: SearchPayload):
  results = answer(payload.query)
  return results

@app.post("/summary")
def search(payload: SearchPayload):
  results = answer(payload.query)
  return results

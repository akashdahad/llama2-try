from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import UnstructuredFileLoader
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from langchain.llms import LlamaCpp
from langchain.llms import CTransformers
from langchain import PromptTemplate, LLMChain
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import torch
import psycopg2
import pdfplumber
import io
import requests
import re
import nltk
import json
import spacy


#? To Start the server
#? python3 -m uvicorn main:app --reload 

app = FastAPI()

# python3 -m spacy download en_core_web_lg To Download Model
nlp = spacy.load("en_core_web_lg")
nltk.download('punkt')


#* Database connection parameters
DBNAME = 'postgres'
USER = 'devadmin'
PASSWORD = 'KCE9MP2En93gLCz2'
HOST = 'bhyve-india-dev-db.postgres.database.azure.com'
PORT = '5432'

#* Connect to the database
connection = psycopg2.connect(dbname=DBNAME, user=USER, password=PASSWORD, host=HOST, port=PORT)
conn = connection.cursor()

model = SentenceTransformer('bert-base-nli-mean-tokens')

# Extract Text
def extract_data_from_file(url):
  text = ''
  contents = requests.get(url)
  with pdfplumber.open(io.BytesIO(contents.content)) as pdf:
    for page in pdf.pages:
      data = page.extract_text(x_tolerance=3, y_tolerance=3)
      if(type(data) == str):
        text = text + ' ' + data
  return { "content": text }

# Extract Text in Array
def extract_data_from_file_array(url):
  arr = []
  contents = requests.get(url)
  with pdfplumber.open(io.BytesIO(contents.content)) as pdf:
    for page in pdf.pages:
      data = page.extract_text(x_tolerance=3, y_tolerance=3)
      arr.append(data)
  return { "content": arr }

# Load Data From Local File
def load_data():
  print("Loading Data")
  loader = UnstructuredFileLoader("./india.txt")
  documents = loader.load()
  return documents

# Split Data into Chunks
def split_data(text, max_chunk_size):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_chunk_size = 0
    for sentence in sentences:
        current_chunk_size += len(sentence)
        if current_chunk_size < max_chunk_size:
            current_chunk.append(sentence)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_chunk_size = len(sentence)
    if current_chunk:
        chunks.append(' '.join(current_chunk))      
    return chunks

# Create Embeddings
def create_embeddings(sentences):
  print("Creating Embeddings")
  embeddings = model.encode(sentences)
  return embeddings

# Store Embeddings
def store_embeddings(content_id, docs, embeddings):
  print("Storing Embeddings")
  for doc, embedding in zip(docs, embeddings):
    st = doc.replace("'", "") 
    conn.execute(f"INSERT INTO pg_embed (content_id, content,  embedding) VALUES ('{content_id}', '{st}', '{embedding.tolist()}')")
    connection.commit()

# Get Search Results
def get_search_results(query):
  query_embeddings = model.encode([query])[0]
  conn.execute(f"SELECT content_id, content FROM pg_embed ORDER BY embedding <-> '{query_embeddings.tolist()}' LIMIT 5")
  results = conn.fetchall()
  return results

# Load LLM
def load_llm():
    llm = LlamaCpp(
    model_path="./models/llama-2-7b-32k-instruct.ggmlv3.q8_0.bin",
    # model_path="./models/llama-2-7b-chat.ggmlv3.q8_0.bin",
    n_gpu_layers=32,
    n_batch=512,
    verbose=True)
    return llm

# Get Answer From LLM
def get_answer_from_llm(template, content, prompt):
  llm = load_llm()
  template = PromptTemplate(template=answer_prompt_template, input_variables=["context", "question"])
  llm_chain = LLMChain(prompt=template, llm=llm)
  result = llm_chain.run(context = content, question = prompt)
  print(result)
  return result

# Pre Process Text
def pre_process(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

# Tag Text with Relevant Skill Tags
def tag_text(input_text, tags):
    processed_text = pre_process(input_text)

    # Using TF-IDF to vectorize the texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_text] + tags)
    
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix).flatten()
    
    # Threshold to consider a tag as relevant can be adjusted
    threshold = 0.15
    relevant_tags = [tags[i] for i, score in enumerate(cosine_similarities[1:]) if score > threshold]
    print("Relevant Tags(0.15): ", relevant_tags)
    return relevant_tags

# Generate Synopsis
def generate_synopsis(text, min, max):
    print("Length:", len(text))
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=max, min_length=min, do_sample=False)
    return summary[0]['summary_text']

# Generate Synopsis Recursively for Large Text
def synopsis_generator(text):
  if(len(text) < 3000):
    return generate_synopsis(text, 100, 300)
  chunks = split_data(text, 3000)
  synopsis = ''
  for chunk in chunks:
    synopsis = synopsis + " " + generate_synopsis(chunk, 30, 100)
    print("Synopsis Length:", len(synopsis))
  if len(synopsis) > 3000:
    print("RE STARTING ---> ", len(synopsis))
    synopsis = synopsis_generator(synopsis)
  else:  
    synopsis = generate_synopsis(synopsis, 100, 300)
  print("FINAL LENGTH ---> ", len(synopsis))
  return synopsis

def summary_generator_per_page(texts):
  summary = []
  for text in texts:
    summary_per_page = synopsis_generator(text)
    summary.append(summary_per_page)
    print(summary_per_page)
    print('---------------------------------------')
  return summary

def get_relevant_skill_tags(url, tags):
  content = extract_data_from_file(url)
  return tag_text(content['content'], tags)

def get_summary_per_page(url):
  content = extract_data_from_file_array(url)
  return summary_generator_per_page(content['content'])

def get_summary(url):
  content = extract_data_from_file(url)
  return synopsis_generator(content['content'])

def get_summary_in_points(url):
  content = extract_data_from_file(url)
  content = content['content'][:80000]
  summary_template = """ You are very intelligent person. Understand the context provided. And Summarize the Context in bullet points. But dont go out of the context. Context : {context}. Summary:  . Return a string which is answer"""
  summary = get_answer_from_llm(summary_template, content, payload.query)
  return summary

def get_synopsis(url):
  content = extract_data_from_file(url)
  return generate_synopsis(content['content'])

class SearchPayload(BaseModel):
    query: str

class LLMPayload(BaseModel):
    template: str
    query: str

class FilePayload(BaseModel):
    file_url: str

class SkillTagsPayload(BaseModel):
    file_url: str
    skill_tags: list

@app.get("/")
def Home():
    return {"Hello": "World"}

@app.post("/search")
def search(payload: SearchPayload):
    results = get_search_results(payload.query)
    return results

@app.post("/skill-tags")
def skill_tags(payload: FilePayload):
    results = get_relevant_skill_tags(payload.file_url, payload.skill_tags)
    return results

@app.post("/summary-per-page")
def summary_per_page(payload: FilePayload):
    results = get_summary_per_page(payload.file_url)
    return results

@app.post("/summary")
def summary(payload: FilePayload):
    results = get_summary(payload.file_url)
    return results

@app.post("/summary-in-points")
def summary_in_points(payload: FilePayload):
  result = get_summary_in_points(payload.file_url)
  return result

@app.post("/synopsis")
def synopsis(payload: FilePayload):
    results = get_synopsis(payload.file_url)
    return results

@app.post("/answer")
def answer(payload: SearchPayload):
    answer_prompt_template = """ You are very intelligent person. Understand the context provided. And answer the following question. But dont go out of the context. Context : {context} Question:  {question} Answer :  . Return a string which is answer"""
    content = ''
    results = get_search_results(payload.query)
    for result in results:
      content = content + ' ' + result[0]
    answer = get_answer_from_llm(answer_prompt_template, content, payload.query)
    return answer

@app.post("/llm")
def llm(payload: LLMPayload):
    content = ''
    results = get_search_results(payload.query)
    for result in results:
      content = content + ' ' + result[0]
    answer = get_answer_from_llm(payload.template, content, payload.query)
    return answer

@app.post("/file/content")
def fileContent(payload: FilePayload):
    result = extract_data_from_file(payload.file_url)
    return result

@app.post("/file/content/storeEmbeddings")
def storeFileEmbeddings(payload: FilePayload):
    result = extract_data_from_file(payload.file_url)
    chunks = split_data(result['content'], 2000)
    embeddings = create_embeddings(chunks)
    #! Update this UUID Dynamically
    result_store = store_embeddings('9fd801e1-aa4c-49dc-bd96-19ab7dbcc8bd', result['content'], embeddings)
    return result_store

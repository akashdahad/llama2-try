# Installation:
# CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

from flask import Flask, request, jsonify
from langchain.llms import LlamaCpp
from langchain.llms import CTransformers
from langchain import PromptTemplate, LLMChain
import torch

app = Flask(__name__)

def load_llm():
    llm = CTransformers(
        model= "models/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        n_gpu_layers=32,
        n_batch=512,
        verbose=True,
    )
    return llm

def get_result(prompt, content):
  llm = load_llm()
  print(prompt, content)
  template = PromptTemplate(template=template, input_variables=["question"])
  llm_chain = LLMChain(prompt=template, llm=llm)
  result = llm_chain.run(prompt)
  print(result)
  return result

@app.route("/")
def home():
    return "Hello, World!"

@app.route('/llm', methods = ['POST'])
def answer():
    body = request.get_json()
    result = get_result(prompt=body['prompt'], content=body['content'])
    return jsonify(result)


def create_app():
   return app


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        print(f"GPU Detected: {device}")
    else:
        print("No GPU detected. Using CPU.")
    
    app.run(host="0.0.0.0", port=5005)

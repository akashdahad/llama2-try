# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="TheBloke/Llama-2-7b-Chat-GGUF")

print(pipe("Hello, I'm a language model, and I love to chat about anything. What do you want to talk about today?"))
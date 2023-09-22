from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# template = """Question: {question}

# Answer: Let's work this out in a step by step way to be sure we have the right answer."""

# prompt = PromptTemplate(template=template, input_variables=["question"])

# # Callbacks support token-wise streaming
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!

llm = LlamaCpp(
    model_path="./models/llama-2-7b-chat.Q2_K.gguf",
    n_gpu_layers=32,
    n_batch=512,
    verbose=True, # Verbose is required to pass to the callback manager
)

prompt = """
Question: A rap battle between Stephen Colbert and John Oliver
"""
print(llm(prompt))
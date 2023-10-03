import vertexai
from vertexai.preview.language_models import TextGenerationModel
from google.oauth2 import service_account
import pdfplumber
import io
import requests


credentials = service_account.Credentials.from_service_account_file('bhyve-project-398118-7220c36439ce.json')
nlp = spacy.load("en_core_web_lg")
nltk.download('punkt')


def get_content():
  text = ''
  contents = requests.get(url)
  with pdfplumber.open(io.BytesIO(contents.content)) as pdf:
    for page in pdf.pages:
      data = page.extract_text(x_tolerance=3, y_tolerance=3)
      if(type(data) == str):
        text = text + ' ' + data
  return { "content": text.lower() }


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


vertexai.init(project="bhyve-project-398118", location="us-central1", credentials=credentials)
parameters = {
    "max_output_tokens": 1024,
    "temperature": 0,
    "top_p": 0.8,
    "top_k": 40
}
model = TextGenerationModel.from_pretrained("text-bison-32k")
response = model.predict(
    """Provide a brief summary for the following article:

LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won\'t cast a spell on him. Daniel Radcliffe as Harry Potter in \"Harry Potter and the Order of the Phoenix\" To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. \"I don\'t plan to be one of those people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar,\" he told an Australian interviewer earlier this month. \"I don\'t think I\'ll be particularly extravagant. \"The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs.\" At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film \"Hostel: Part II,\" currently six places below his number one movie on the UK box office chart. Details of how he\'ll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. \"I\'ll definitely have some sort of party,\" he said in an interview. \"Hopefully none of you will be reading about it.\" Radcliffe\'s earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. \"People are always looking to say \'kid star goes off the rails,\'\" he told reporters last month. \"But I try very hard not to go that way because it would be too easy for them.\" His latest outing as the boy wizard in \"Harry Potter and the Order of the Phoenix\" is breaking records on both sides of the Atlantic and he will reprise the role in the last two films. Watch I-Reporter give her review of Potter\'s latest » . There is life beyond Potter, however. The Londoner has filmed a TV movie called \"My Boy Jack,\" about author Rudyard Kipling and his son, due for release later this year. He will also appear in \"December Boys,\" an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer\'s \"Equus.\" Meanwhile, he is braced for even closer media scrutiny now that he\'s legally an adult: \"I just think I\'m going to be more sort of fair game,\" he told Reuters. E-mail to a friend . Copyright 2007 Reuters. All rights reserved.This material may not be published, broadcast, rewritten, or redistributed.""",
    **parameters
)
print(f"Response from Model: {response.text}")
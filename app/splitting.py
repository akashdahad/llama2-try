import pdfplumber
import io
import requests
import re
import nltk
from nltk.tokenize import sent_tokenize
import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


# python3 -m spacy download en_core_web_sm To Download Model
nlp = spacy.load("en_core_web_lg")
nltk.download('punkt')

def extract_data_from_file(url):
  text = ''
  try:
    contents = requests.get(url)
    with pdfplumber.open(io.BytesIO(contents.content)) as pdf:
      for page in pdf.pages:
        data = page.extract_text(x_tolerance=3, y_tolerance=3)
        if(type(data) == str):
          text = text + ' ' + data
    return text
  except:
    return text

def read_json_file(file_name):
  with open(file_name) as f:
    data = json.load(f)
  return data

def chunk_text(text, max_chunk_size):
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


def pre_process(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

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
    # threshold = 0.2
    # relevant_tags = [tags[i] for i, score in enumerate(cosine_similarities[1:]) if score > threshold]    
    # print("Relevant Tags(0.2): ", relevant_tags)
    return relevant_tags

def generate_synopsis(text):
    summarizer = pipeline("summarization", model="ainize/bart-base-cnn")
    summary = summarizer(text, max_length=250, min_length=30, do_sample=False)
    print(len(summary[0]['summary_text']))
    return summary[0]['summary_text']

def synopsis_generator(text):
  chunks = chunk_text(text, 3000)
  synopsis = ''
  for chunk in chunks:
    synopsis = synopsis + " " + generate_synopsis(chunk)
    print(len(synopsis))
  if len(synopsis) > 3000:
    print("RE STARTING ---> ", len(synopsis))
    synopsis = synopsis_generator(synopsis)
  else:  
    synopsis = generate_synopsis(synopsis)
  print("FINAL LENGTH ---> ", len(synopsis))
  return synopsis

data_map = read_json_file('data.json')
skills_map = read_json_file('skills.json')

skills = []

# >>>> EXAMPLE -> SKILL TAG GENERATION

# for skill in skills_map:
#   skills.append(skill['displayName'])

# for item in data_map:
#   file = item['uploadedFileUrl']
#   skill = item['displayNames']
#   print("Original Skill: ", skill)
#   text = extract_data_from_file(file)
#   new_skills =  tag_text(text, skills)


# >>>> EXAMPLE -> SYNOPSIS GENERATION

# Extract Text
# Check if Length > 3,000 Characters
# Split Text into Chunks of 3,000 Characters
# Generate Synopsis for Each Chunk
# Check if Total Length of All Synopsis > 3,000 Characters
# If Yes, Combine Synopsis into One
# Else, Split into Chunks of 3,000 Characters
# Generate Synopsis for Each Chunk
# Check if Total Length of All Synopsis > 3,000 Characters
# If Yes, Combine Synopsis into One 
# Generate Final Synopsis

# for item in data_map:
#   file = item['uploadedFileUrl']
#   text = extract_data_from_file(file)
#   print(len(text))
#   synopsis = synopsis_generator(text)
#   print(synopsis)

file = data_map[17]['uploadedFileUrl']
text = extract_data_from_file(file)
print((text))
synopsis = synopsis_generator(text)
print('---------------------------------------')
print(synopsis)
print('---------------------------------------')

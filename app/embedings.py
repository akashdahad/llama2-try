from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('bert-base-nli-mean-tokens')
embeddings = model.encode(sentences)
print(embeddings)
print(len(embeddings[0].tolist()))

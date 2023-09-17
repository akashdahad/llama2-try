import psycopg2
from sentence_transformers import SentenceTransformer

# Database connection parameters
DBNAME = 'postgres'
USER = 'devadmin'
PASSWORD = 'KCE9MP2En93gLCz2'
HOST = 'bhyve-india-dev-db.postgres.database.azure.com'
PORT = '5432'

# Connect to the database
connection = psycopg2.connect(dbname=DBNAME, user=USER, password=PASSWORD, host=HOST, port=PORT)
conn = connection.cursor()

# conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
# conn.execute('DROP TABLE IF EXISTS pg_embed')
# conn.execute('CREATE TABLE pg_embed (id bigserial PRIMARY KEY, content text, content_id text, embedding vector(768))')
# connection.commit()

inputs = [
    'The dog is barking',
    'The cat is purring',
    'The bear is growling'
]

# model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('bert-base-nli-mean-tokens')
embeddings = model.encode(inputs)

print(embeddings)

# for content, embedding in zip(inputs, embeddings):
#     print(content, embedding)
#     conn.execute(f"INSERT INTO pg_embed (content_id, content,  embedding) VALUES ('content1', '{content}', '{embedding.tolist()}')")
#     connection.commit()
    # INSERT INTO items (embedding) VALUES ('[1,2,3]'), ('[4,5,6]');
    # conn.execute(f"INSERT INTO pg_embed (content_id, content, embedding) VALUES ('1', 'hi', '[1,2,3]')")

query = 'The dog is barking'
embedding_query = model.encode([query])[0]
conn.execute(f"SELECT content FROM pg_embed ORDER BY embedding <-> '{embedding_query.tolist()}' LIMIT 1")
results = conn.fetchall()
print(results)

conn.close()
connection.close()
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

query = 'indian geography different from china'

model = SentenceTransformer('bert-base-nli-mean-tokens')
query_embeddings = model.encode([query])[0]

conn.execute(f"SELECT content FROM pg_embed ORDER BY embedding <-> '{query_embeddings.tolist()}' LIMIT 10")
results = conn.fetchall()

for result in results:
  print(result)

conn.close()
connection.close()
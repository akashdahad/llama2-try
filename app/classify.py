import nltk
import gensim
from nltk.corpus import stopwords
from gensim import corpora

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return tokens

def classify_into_parts(text, num_topics=5):
    # Preprocess the text
    tokens = preprocess(text)

    # Create a dictionary and corpus required for LDA
    dictionary = corpora.Dictionary([tokens])
    corpus = [dictionary.doc2bow(tokens)]

    # Apply LDA
    lda = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    
    topics = lda.print_topics(num_words=5)
    
    return topics

text = """Your long text goes here. This could be a document that covers multiple topics.
          For example, it might discuss the history of the Roman Empire, delve into the intricacies
          of molecular biology, and provide a critique on modern art movements."""

topics = classify_into_parts(text, num_topics=3)
for idx, topic in topics:
    print(f"Topic {idx + 1}: {topic}\n")

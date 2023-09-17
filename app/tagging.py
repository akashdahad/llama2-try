from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def tag_content_with_relevant_skills(content, skill_tags):
    # Tokenize and preprocess the content and skill tags
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Create vectors for content and skill tags
    all_texts = [content] + skill_tags
    vectors = vectorizer.fit_transform(all_texts)
    
    # Get the vector for the content (it's the first one)
    content_vector = vectors[0]
    
    # Calculate the cosine similarities between the content and each skill tag
    cosine_similarities = linear_kernel(content_vector, vectors).flatten()
    
    # Sort the skill tags based on the similarity scores
    related_skill_tags_indices = cosine_similarities.argsort()[:-len(all_texts)-1:-1]
    
    # Get the most related skill tags (excluding the content itself)
    most_related_skill_tags = [skill_tags[i-1] for i in related_skill_tags_indices[1:]]
    
    return most_related_skill_tags

# Example:
content = "I am proficient in Python, data analysis, and machine learning."
skill_tags = ["Python", "JavaScript", "Data Analysis", "Web Development", "Machine Learning"]
print(tag_content_with_relevant_skills(content, skill_tags))

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)   # remove special characters
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    return text

# Take user input
resume_text = input("Paste your Resume text:\n")
job_description = input("\nPaste Job Description:\n")

# Clean text
resume_text = clean_text(resume_text)
job_description = clean_text(job_description)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform([resume_text, job_description])

# Cosine similarity
similarity = cosine_similarity(vectors[0], vectors[1])
match_percentage = similarity[0][0] * 100

print("\n🔎 Match Result:")
print(f"✅ Resume matches job description by {match_percentage:.2f}%")
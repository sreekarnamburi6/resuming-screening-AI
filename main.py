import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Predefined skill list
skills_list = [
    "python", "machine learning", "deep learning",
    "nlp", "flask", "opencv", "data analysis",
    "tensorflow", "pandas", "numpy"
]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_skills(text):
    found_skills = []
    for skill in skills_list:
        if skill in text:
            found_skills.append(skill)
    return found_skills

# Read resume from file
with open("resume.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()

# Read job description from file
with open("job.txt", "r", encoding="utf-8") as f:
    job_description = f.read()

# Clean text
resume_text = clean_text(resume_text)
job_description = clean_text(job_description)

if not resume_text.strip() or not job_description.strip():
    print("Error: resume.txt or job.txt is empty.")
    exit()

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform([resume_text, job_description])
similarity = cosine_similarity(vectors[0], vectors[1])
match_percentage = similarity[0][0] * 100

# Skill Extraction
resume_skills = extract_skills(resume_text)
job_skills = extract_skills(job_description)
missing_skills = list(set(job_skills) - set(resume_skills))

print("\n🔎 Match Result:")
print(f"✅ Resume matches job description by {match_percentage:.2f}%")

print("\n🛠 Skills Found in Resume:")
print(resume_skills)

print("\n❌ Missing Skills:")
print(missing_skills)

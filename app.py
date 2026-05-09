import re
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Predefined skills
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

@app.route("/", methods=["GET", "POST"])
def index():
    match_percentage = None
    resume_skills = []
    missing_skills = []

    if request.method == "POST":
        resume_text = request.form["resume"]
        job_description = request.form["job"]

        resume_text = clean_text(resume_text)
        job_description = clean_text(job_description)

        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform([resume_text, job_description])
        similarity = cosine_similarity(vectors[0], vectors[1])
        match_percentage = round(similarity[0][0] * 100, 2)

        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_description)
        missing_skills = list(set(job_skills) - set(resume_skills))

    return render_template("index.html",
                           match_percentage=match_percentage,
                           resume_skills=resume_skills,
                           missing_skills=missing_skills)

if __name__ == "__main__":
    app.run(debug=True)
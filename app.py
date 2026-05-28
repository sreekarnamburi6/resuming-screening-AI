import re
from flask import Flask, render_template, request, jsonify, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
print("API KEY:", os.getenv("OPENAI_API_KEY"))

client = None
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    print("OpenAI client initialization error:", e)

app = Flask(__name__)

skills_list = [
    "python", "machine learning", "deep learning",
    "nlp", "flask", "opencv", "data analysis",
    "tensorflow", "pandas", "numpy"
]

role_keywords = [
    "data analyst", "machine learning engineer", "ai engineer",
    "data scientist", "software engineer", "backend developer",
    "frontend developer", "python developer", "full stack developer"
]

impact_keywords = [
    "increased", "reduced", "improved", "achieved",
    "optimized", "generated", "boosted", "led", "managed"
]

generic_words = [
    "hardworking", "passionate", "dedicated",
    "quick learner", "self motivated",
    "team player", "enthusiastic"
]

growth_keywords = [
    "led", "managed", "senior", "head", "lead",
    "promoted", "advanced", "progressed", "growth",
    "increased responsibility", "team lead", "supervised"
]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_skills(text):
    return [skill for skill in skills_list if skill in text]

def extract_pdf_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def rewrite_with_ai(resume_text, job_description):
    prompt = f"""
You are a professional ATS resume optimizer.

Rewrite the following resume to better match the job description.

Rules:
- Do NOT fabricate fake experience.
- Do NOT add false metrics.
- Improve clarity and structure.
- Insert missing relevant keywords naturally.
- Keep it professional and concise.
- Output clean resume text only (no explanations).

Resume:
{resume_text}

Job Description:
{job_description}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert resume writer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        if response and hasattr(response, 'choices') and len(response.choices) > 0:
            return response.choices[0].message.content
        return None

    except Exception as e:
        print("Error calling AI rewrite:", e)
        return None

@app.route("/analyze", methods=["POST"])
def analyze():

    job_description = request.form["job"]

    pdf_file = request.files.get("resume_pdf")
    if pdf_file and pdf_file.filename != "":
        resume_text = extract_pdf_text(pdf_file)
    else:
        resume_text = request.form.get("resume", "")

    resume_text = clean_text(resume_text)
    job_description = clean_text(job_description)

    # -------- Improved ATS Keyword Coverage --------

    # Use TF-IDF to extract important keywords from job description
    vectorizer_ats = TfidfVectorizer(stop_words='english', max_features=20)
    tfidf_matrix = vectorizer_ats.fit_transform([job_description])
    job_keywords = vectorizer_ats.get_feature_names_out()

    resume_words = set(resume_text.split())

    matched_keywords = [word for word in job_keywords if word in resume_words]

    if len(job_keywords) > 0:
        ats_score = round((len(matched_keywords) / len(job_keywords)) * 100, 2)
    else:
        ats_score = 0

    if ats_score >= 70:
        ats_feedback = "Strong ATS keyword alignment."
    elif ats_score >= 50:
        ats_feedback = "Moderate ATS alignment. Improve keyword matching."
    else:
        ats_feedback = "Low ATS alignment. Add more relevant job keywords."

    # Return which important ATS keywords are matched / missing
    missing_ats_keywords = list(set(job_keywords) - set(matched_keywords))

    role_detected = None
    for role in role_keywords:
        if role in resume_text:
            role_detected = role
            break

    if role_detected:
        role_clarity = f"✅ Clear role detected: {role_detected.title()}"
    else:
        role_clarity = "❌ No clear target role detected. Add a specific role title in your resume headline."

    # -------- Impact Detection --------
    has_numbers = bool(re.search(r'\d+', resume_text))
    has_percentage = "%" in resume_text

    impact_verbs_found = [word for word in impact_keywords if word in resume_text]

    if has_numbers or has_percentage or impact_verbs_found:
        impact_score = "✅ Impact evidence detected (metrics/results found)."
    else:
        impact_score = "❌ No measurable results found. Add numbers, percentages, or achievements."

    # -------- Generic Summary Detection --------
    generic_found = [word for word in generic_words if word in resume_text]

    if generic_found:
        summary_feedback = "⚠ Generic terms detected in summary: " + ", ".join(generic_found) + ". Add specific skills and measurable impact instead."
    else:
        summary_feedback = "✅ No generic buzzwords detected. Summary appears focused."

    # -------- Growth & Progression Detection --------
    growth_found = [word for word in growth_keywords if word in resume_text]

    if growth_found:
        growth_feedback = "✅ Growth indicators detected (leadership/progression)."
    else:
        growth_feedback = "⚠ Limited growth indicators. Add leadership or progression evidence."

    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(vectors[0], vectors[1])
    similarity_score = round(similarity[0][0] * 100, 2)

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)

    matched_skills = list(set(resume_skills) & set(job_skills))
    missing_skills = list(set(job_skills) - set(resume_skills))

    skill_score = round((len(matched_skills) / len(job_skills)) * 100, 2) if job_skills else 0
    match_percentage = round((0.6 * skill_score) + (0.4 * similarity_score), 2)

    # -------- HR 6-Second Score --------

    hr_score = 0

    # Role clarity
    if "✅" in role_clarity:
        hr_score += 20

    # Impact
    if "✅" in impact_score:
        hr_score += 20

    # ATS
    if ats_score >= 70:
        hr_score += 25
    elif ats_score >= 50:
        hr_score += 15
    else:
        hr_score += 5

    # Growth
    if "✅" in growth_feedback:
        hr_score += 15

    # Summary
    if "✅" in summary_feedback:
        hr_score += 10

    # Skill score
    if skill_score >= 70:
        hr_score += 10
    else:
        hr_score += 5

    hr_score = min(hr_score, 100)

    return jsonify({
        "match_percentage": match_percentage,
        "similarity_score": similarity_score,
        "skill_score": skill_score,
        "resume_skills": resume_skills,
        "missing_skills": missing_skills,
        "role_clarity": role_clarity,
        "impact_score": impact_score,
        "summary_feedback": summary_feedback,
        "ats_score": ats_score,
        "ats_feedback": ats_feedback,
        "matched_ats_keywords": matched_keywords,
        "missing_ats_keywords": missing_ats_keywords,
        "growth_feedback": growth_feedback
        ,"hr_score": hr_score
    })

@app.route("/download-report", methods=["POST"])
def download_report():

    data = request.json

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 40

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, y, "Resume Analysis Report")
    y -= 30

    pdf.setFont("Helvetica", 11)

    for key, value in data.items():
        if isinstance(value, list):
            pdf.drawString(40, y, f"{key}:")
            y -= 20
            for item in value:
                pdf.drawString(60, y, f"- {item}")
                y -= 15
        else:
            pdf.drawString(40, y, f"{key}: {value}")
            y -= 20

        if y < 50:
            pdf.showPage()
            pdf.setFont("Helvetica", 11)
            y = height - 40

    pdf.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="Resume_Analysis_Report.pdf",
        mimetype='application/pdf'
    )


@app.route("/rewrite", methods=["POST"])
def rewrite_resume():
    resume_text = request.json.get("resume")
    job_description = request.json.get("job")

    if not resume_text or not job_description:
        return jsonify({"error": "Missing 'resume' or 'job' fields."}), 400

    if client is None:
        return jsonify({"error": "OpenAI API key not configured on server."}), 500

    improved_resume = rewrite_with_ai(resume_text, job_description)
    if improved_resume is None:
        return jsonify({"error": "AI rewrite failed."}), 500

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 40
    pdf.setFont("Helvetica", 11)

    for line in improved_resume.split("\n"):
        pdf.drawString(40, y, line)
        y -= 15
        if y < 40:
            pdf.showPage()
            pdf.setFont("Helvetica", 11)
            y = height - 40

    pdf.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="Optimized_Resume.pdf",
        mimetype="application/pdf"
    )

@app.route("/", methods=["GET", "POST"])
def index():
    match_percentage = None
    similarity_score = None
    skill_score = None
    resume_skills = []
    missing_skills = []

    if request.method == "POST":

        job_description = request.form["job"]

        # If PDF uploaded
        if "resume_pdf" in request.files:
            pdf_file = request.files["resume_pdf"]
            if pdf_file.filename != "":
                resume_text = extract_pdf_text(pdf_file)
            else:
                resume_text = request.form["resume"]
        else:
            resume_text = request.form["resume"]

        optimize = request.form.get("optimize")

        resume_text = clean_text(resume_text)
        job_description = clean_text(job_description)

        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform([resume_text, job_description])
        similarity = cosine_similarity(vectors[0], vectors[1])
        similarity_score = round(similarity[0][0] * 100, 2)

        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_description)

        matched_skills = list(set(resume_skills) & set(job_skills))
        missing_skills = list(set(job_skills) - set(resume_skills))

        if len(job_skills) > 0:
            skill_score = round((len(matched_skills) / len(job_skills)) * 100, 2)
        else:
            skill_score = 0

        match_percentage = round((0.6 * skill_score) + (0.4 * similarity_score), 2)

        improvement_tips = []
        if optimize:
            if missing_skills:
                for skill in missing_skills:
                    improvement_tips.append(f"Consider learning and adding '{skill}' to your resume.")

            if match_percentage is not None:
                if match_percentage < 50:
                    improvement_tips.append("Your resume needs significant improvement for this role.")
                elif match_percentage < 70:
                    improvement_tips.append("You are moderately aligned. Add missing skills and relevant projects.")
                else:
                    improvement_tips.append("Your resume is well aligned with this job description.")
    else:
        improvement_tips = []

    return render_template("index.html",
                           match_percentage=match_percentage,
                           similarity_score=similarity_score,
                           skill_score=skill_score,
                           resume_skills=resume_skills,
                           missing_skills=missing_skills,
                           improvement_tips=improvement_tips)

if __name__ == "__main__":
    app.run(debug=True)

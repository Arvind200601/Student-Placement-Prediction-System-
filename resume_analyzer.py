# ==========================================================
# Student Placement Prediction System using Machine Learning
# Professional Resume Analyzer
# Developed for Resume Screening & Placement Prediction
# ==========================================================

import re
import os
import pdfplumber
import PyPDF2

# ==========================================================
# TECHNICAL SKILLS DATABASE
# ==========================================================

SKILLS_DATABASE = [

    # Programming Languages
    "python",
    "java",
    "c",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "php",
    "ruby",
    "go",
    "kotlin",
    "swift",
    "r",

    # Web Development
    "html",
    "css",
    "bootstrap",
    "tailwind",
    "react",
    "angular",
    "vue",
    "node",
    "express",
    "flask",
    "django",

    # Database
    "mysql",
    "postgresql",
    "mongodb",
    "sqlite",
    "oracle",
    "firebase",

    # Data Science & ML
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "data science",
    "pandas",
    "numpy",
    "matplotlib",
    "opencv",
    "tensorflow",
    "keras",
    "pytorch",
    "scikit-learn",

    # Cloud
    "aws",
    "azure",
    "gcp",

    # DevOps
    "docker",
    "kubernetes",
    "git",
    "github",
    "jenkins",

    # Networking
    "ccna",
    "networking",
    "computer networks",

    # Core Subjects
    "dbms",
    "operating system",
    "os",
    "oops",
    "data structures",
    "algorithms",

    # Soft Skills
    "communication",
    "leadership",
    "teamwork",
    "problem solving",
    "critical thinking",
    "presentation"
]

# ==========================================================
# PDF TEXT EXTRACTION
# ==========================================================

def extract_text_from_pdf(pdf_path):

    text = ""

    try:

        with pdfplumber.open(pdf_path) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:

                    text += page_text + "\n"

    except Exception:

        try:

            with open(pdf_path, "rb") as file:

                reader = PyPDF2.PdfReader(file)

                for page in reader.pages:

                    page_text = page.extract_text()

                    if page_text:

                        text += page_text + "\n"

        except Exception:

            return ""

    return text.lower()
# ==========================================================
# NAME EXTRACTION
# ==========================================================

def extract_name(text):

    lines = text.split("\n")

    for line in lines[:10]:

        line = line.strip()

        if len(line) > 3 and len(line) < 40:

            if not any(char.isdigit() for char in line):

                if "resume" not in line and "curriculum" not in line:

                    return line.title()

    return "Unknown Candidate"


# ==========================================================
# EMAIL EXTRACTION
# ==========================================================

def extract_email(text):

    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

    match = re.search(pattern, text)

    if match:

        return match.group()

    return "Not Found"


# ==========================================================
# PHONE NUMBER EXTRACTION
# ==========================================================

def extract_phone(text):

    pattern = r"(?:\+91[-\s]?)?[6-9]\d{9}"

    match = re.search(pattern, text)

    if match:

        return match.group()

    return "Not Found"


# ==========================================================
# LINKEDIN EXTRACTION
# ==========================================================

def extract_linkedin(text):

    pattern = r"(https?:\/\/)?(www\.)?linkedin\.com\/[A-Za-z0-9\/\-\_]+"

    match = re.search(pattern, text)

    if match:

        return match.group()

    return "Not Available"


# ==========================================================
# GITHUB EXTRACTION
# ==========================================================

def extract_github(text):

    pattern = r"(https?:\/\/)?(www\.)?github\.com\/[A-Za-z0-9\-\_]+"

    match = re.search(pattern, text)

    if match:

        return match.group()

    return "Not Available"


# ==========================================================
# CLEAN RESUME TEXT
# ==========================================================

def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    text = text.replace("|", " ")

    text = text.replace(",", " ")

    text = text.replace(";", " ")

    text = text.lower()

    return text
# ==========================================================
# SKILL EXTRACTION
# ==========================================================

def extract_skills(text):

    found_skills = []

    text = clean_text(text)

    for skill in SKILLS_DATABASE:

        if skill.lower() in text:

            found_skills.append(skill.title())

    return sorted(list(set(found_skills)))


# ==========================================================
# PROJECT DETECTION
# ==========================================================

def count_projects(text):

    keywords = [

        "project",

        "projects",

        "developed",

        "built",

        "created",

        "implementation",

        "application",

        "system"

    ]

    count = 0

    text = clean_text(text)

    for word in keywords:

        count += text.count(word)

    if count > 8:

        count = 8

    return count


# ==========================================================
# INTERNSHIP DETECTION
# ==========================================================

def count_internships(text):

    keywords = [

        "internship",

        "intern",

        "industrial training",

        "summer internship",

        "winter internship",

        "worked as"

    ]

    count = 0

    text = clean_text(text)

    for word in keywords:

        count += text.count(word)

    if count > 5:

        count = 5

    return count


# ==========================================================
# CERTIFICATION DETECTION
# ==========================================================

def count_certifications(text):

    keywords = [

        "certificate",

        "certification",

        "certified",

        "coursera",

        "udemy",

        "infosys",

        "nptel",

        "aws",

        "google",

        "microsoft",

        "oracle",

        "ibm"

    ]

    count = 0

    text = clean_text(text)

    for word in keywords:

        count += text.count(word)

    if count > 10:

        count = 10

    return count


# ==========================================================
# RESUME SCORE CALCULATION
# ==========================================================

def calculate_resume_score(skills,
                           projects,
                           internships,
                           certifications):

    score = 0

    # Skills (40 Marks)

    score += min(len(skills) * 2, 40)

    # Projects (20 Marks)

    score += min(projects * 4, 20)

    # Internship (20 Marks)

    score += min(internships * 10, 20)

    # Certifications (20 Marks)

    score += min(certifications * 2, 20)

    if score > 100:

        score = 100

    return score
# ==========================================================
# AI SUGGESTIONS
# ==========================================================

def generate_ai_suggestions(score,
                            skills,
                            projects,
                            internships,
                            certifications):

    suggestions = []

    if score >= 85:

        suggestions.append("Excellent Resume. You are highly placement ready.")

    if len(skills) < 8:

        suggestions.append("Learn more technical skills such as Python, SQL, Machine Learning and Cloud Computing.")

    if projects < 3:

        suggestions.append("Build at least 3 real-world projects and upload them to GitHub.")

    if internships == 0:

        suggestions.append("Complete at least one internship to improve your industry exposure.")

    if certifications < 3:

        suggestions.append("Earn more certifications from Coursera, NPTEL, Infosys or Google.")

    if "communication" not in [s.lower() for s in skills]:

        suggestions.append("Improve communication and presentation skills.")

    if score < 60:

        suggestions.append("Practice aptitude, coding and interview questions regularly.")

    if len(suggestions) == 0:

        suggestions.append("Your profile is excellent. Keep updating your resume.")

    return suggestions


# ==========================================================
# STRENGTHS
# ==========================================================

def get_strengths(score,
                  skills,
                  projects,
                  internships):

    strengths = []

    if score >= 80:
        strengths.append("Excellent Overall Resume")

    if len(skills) >= 10:
        strengths.append("Strong Technical Skills")

    if projects >= 3:
        strengths.append("Good Project Experience")

    if internships >= 1:
        strengths.append("Industry Exposure")

    if len(strengths) == 0:
        strengths.append("Good Learning Potential")

    return strengths


# ==========================================================
# WEAK AREAS
# ==========================================================

def get_weak_areas(score,
                   skills,
                   projects,
                   internships,
                   certifications):

    weak = []

    if score < 70:
        weak.append("Overall Resume Quality")

    if len(skills) < 8:
        weak.append("Technical Skills")

    if projects < 3:
        weak.append("Project Experience")

    if internships == 0:
        weak.append("Internship Experience")

    if certifications < 3:
        weak.append("Professional Certifications")

    return weak


# ==========================================================
# ELIGIBLE COMPANIES
# ==========================================================

def recommend_companies(score):

    if score >= 90:

        return [

            "Google",

            "Microsoft",

            "Amazon",

            "Adobe",

            "Atlassian",

            "ServiceNow"

        ]

    elif score >= 80:

        return [

            "Infosys",

            "TCS",

            "Accenture",

            "Cognizant",

            "Capgemini",

            "Zoho"

        ]

    elif score >= 65:

        return [

            "Wipro",

            "HCL",

            "Tech Mahindra",

            "LTIMindtree",

            "Hexaware"

        ]

    else:

        return [

            "Improve Resume to Become Eligible"

        ]


# ==========================================================
# RECOMMENDED COURSES
# ==========================================================

def recommend_courses(score):

    courses = [

        "Python Programming",

        "Data Structures & Algorithms",

        "SQL & Database Management",

        "Machine Learning",

        "Web Development",

        "Cloud Computing (AWS)",

        "Git & GitHub",

        "Interview Preparation",

        "Aptitude & Logical Reasoning",

        "Communication Skills"

    ]

    return courses


# ==========================================================
# PLACEMENT READINESS
# ==========================================================

def placement_level(score):

    if score >= 90:
        return "★★★★★ Excellent"

    elif score >= 80:
        return "★★★★ Very Good"

    elif score >= 70:
        return "★★★ Good"

    elif score >= 60:
        return "★★ Average"

    else:
        return "★ Needs Improvement"
    # ==========================================================
# MAIN RESUME ANALYZER FUNCTION
# ==========================================================

def analyze_resume(pdf_path):

    # -----------------------------------------
    # Extract Text
    # -----------------------------------------

    text = extract_text_from_pdf(pdf_path)

    if text.strip() == "":
        return {
            "candidate_name": "Unknown Candidate",
            "email": "Not Found",
            "phone": "Not Found",
            "linkedin": "Not Available",
            "github": "Not Available",
            "skills": [],
            "projects": 0,
            "internship": 0,
            "certifications": 0,
            "resume_score": 0,
            "strengths": [],
            "weak_areas": ["Unable to read resume"],
            "suggestions": ["Please upload a valid PDF resume."],
            "companies": [],
            "courses": [],
            "placement_level": "Needs Improvement"
        }

    # -----------------------------------------
    # Candidate Details
    # -----------------------------------------

    candidate_name = extract_name(text)

    email = extract_email(text)

    phone = extract_phone(text)

    linkedin = extract_linkedin(text)

    github = extract_github(text)

    # -----------------------------------------
    # Resume Analysis
    # -----------------------------------------

    skills = extract_skills(text)

    projects = count_projects(text)

    internship = count_internships(text)

    certifications = count_certifications(text)

    resume_score = calculate_resume_score(
        skills,
        projects,
        internship,
        certifications
    )

    # -----------------------------------------
    # AI Suggestions
    # -----------------------------------------

    suggestions = generate_ai_suggestions(
        resume_score,
        skills,
        projects,
        internship,
        certifications
    )

    strengths = get_strengths(
        resume_score,
        skills,
        projects,
        internship
    )

    weak_areas = get_weak_areas(
        resume_score,
        skills,
        projects,
        internship,
        certifications
    )

    companies = recommend_companies(
        resume_score
    )

    courses = recommend_courses(
        resume_score
    )

    placement = placement_level(
        resume_score
    )

    # -----------------------------------------
    # Final Report
    # -----------------------------------------

    report = {

        "candidate_name": candidate_name,

        "email": email,

        "phone": phone,

        "linkedin": linkedin,

        "github": github,

        "skills": skills,

        "projects": projects,

        "internship": internship,

        "certifications": certifications,

        "resume_score": resume_score,

        "strengths": strengths,

        "weak_areas": weak_areas,

        "suggestions": suggestions,

        "companies": companies,

        "courses": courses,

        "placement_level": placement

    }

    return report
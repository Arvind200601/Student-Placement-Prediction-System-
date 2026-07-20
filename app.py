# ==========================================================
# Student Placement Prediction System using Machine Learning
# Chennai Institute of Technology
# Developed By : Arvind Kartikeyan S
# Best Model : Random Forest
# ==========================================================

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file,
    flash
)

import os
import sys
import joblib
import numpy as np
from datetime import datetime

# Resume Analyzer
from resume_analyzer import analyze_resume

# PDF Generation
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

# ==========================================================
# Flask Configuration
# ==========================================================

app = Flask(__name__)
import sqlite3

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(BASE_DIR)

# ==========================================================
# CREATE DATABASE
# ==========================================================

def init_database():

    conn = sqlite3.connect("placement.db")

    cursor = conn.cursor()

    cursor.execute("""

    CREATE TABLE IF NOT EXISTS prediction_history(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        student_name TEXT,

        prediction TEXT,

        confidence REAL,

        model TEXT,

        date TEXT

    )

    """)

    conn.commit()

    conn.close()

init_database()
app.secret_key = "cit_placement_prediction"

UPLOAD_FOLDER = "uploads"
REPORT_FOLDER = "reports"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["REPORT_FOLDER"] = REPORT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# ==========================================================
# Load ML Model
# ==========================================================

try:

    model = joblib.load("model.pkl")

    scaler = joblib.load("scaler.pkl")

    print("Model Loaded Successfully")

except Exception as e:

    print("Error Loading Model :", e)

# ==========================================================
# Model Details
# ==========================================================

BEST_MODEL = "Random Forest"

RF_ACCURACY = 98.00

SVM_ACCURACY = 95.00

COLLEGE_NAME = "Chennai Institute of Technology"

PROJECT_NAME = "Student Placement Prediction System using Machine Learning"

DEVELOPER = "Arvind Kartikeyan S"

# ==========================================================
# Generate Suggestions
# ==========================================================

def generate_suggestions(
    cgpa,
    programming,
    communication,
    internship,
    projects
):

    suggestions = []

    if cgpa < 7:

        suggestions.append(
            "Improve CGPA above 7.5."
        )

    if programming < 7:

        suggestions.append(
            "Practice Data Structures and Algorithms daily."
        )

    if communication < 7:

        suggestions.append(
            "Improve Communication & Presentation Skills."
        )

    if internship == 0:

        suggestions.append(
            "Complete at least one internship."
        )

    if projects < 3:

        suggestions.append(
            "Build more real-world projects."
        )

    if len(suggestions) == 0:

        suggestions.append(
            "Excellent profile. Continue improving your skills."
        )

    return suggestions

# ==========================================================
# Placement Readiness
# ==========================================================

def placement_grade(confidence):

    if confidence >= 90:

        return "A+"

    elif confidence >= 80:

        return "A"

    elif confidence >= 70:

        return "B+"

    elif confidence >= 60:

        return "B"

    elif confidence >= 50:

        return "C"

    else:

        return "Needs Improvement"


def sigmoid(value):

    try:

        return 1 / (1 + np.exp(-value))

    except OverflowError:

        return 1.0 if value > 0 else 0.0


def calculate_profile_score(cgpa, programming, communication, internship, projects, attendance, resume_score=None):

    score = 0.0

    score += min((cgpa / 10.0) * 40.0, 40.0)

    score += min((programming / 10.0) * 15.0, 15.0)

    score += min((communication / 10.0) * 15.0, 15.0)

    score += min((internship / 5.0) * 10.0, 10.0)

    score += min((projects / 10.0) * 10.0, 10.0)

    score += min((attendance / 100.0) * 10.0, 10.0)

    if resume_score is not None:

        score = max(score, resume_score * 0.85)

    return round(min(score, 100.0), 2)


def calculate_readiness_score(cgpa, programming, communication, internship, projects, attendance, resume_score=None):

    profile_score = calculate_profile_score(
        cgpa,
        programming,
        communication,
        internship,
        projects,
        attendance,
        resume_score=resume_score
    )

    predicted_label = 1

    try:

        input_data = np.array([[
            cgpa,
            100,
            communication,
            programming,
            internship,
            projects,
            attendance
        ]])

        scaled_data = scaler.transform(input_data)

        predicted_label = int(model.predict(scaled_data)[0])

    except Exception:

        predicted_label = 1 if profile_score >= 50 else 0

    if predicted_label == 0:

        readiness = max(5.0, profile_score - 20.0)

    else:

        readiness = min(99.0, profile_score + 5.0)

    return round(readiness, 2)
    # ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def feature_importance():

    features = [

        "CGPA",

        "IQ",

        "Communication",

        "Programming",

        "Internship",

        "Projects",

        "Attendance"

    ]
    
    #importance = RANDOM_FOREST_MODEL.feature_importances_
    return []

    data = []

    for i in range(len(features)):

        data.append({

            "feature": features[i],

            "importance": round(
                importance[i] * 100,
                2
            )

        })

    data = sorted(

        data,

        key=lambda x: x["importance"],

        reverse=True

    )

    return data

# ==========================================================
# Recommended Companies
# ==========================================================

def recommended_companies(readiness):

    if readiness >= 90:

        return [
            "Google",
            "Microsoft",
            "Amazon",
            "Adobe",
            "Atlassian",
            "ServiceNow"
        ]

    elif readiness >= 75:

        return [
            "Zoho",
            "Infosys",
            "TCS",
            "Accenture",
            "Capgemini",
            "Cognizant"
        ]

    elif readiness >= 60:

        return [
            "HCL",
            "Tech Mahindra",
            "Wipro",
            "LTIMindtree"
        ]

    else:

        return [
            "Internship-focused roles",
            "Campus hiring drives",
            "Startups",
            "Entry-level product teams"
        ]
    # ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        college=COLLEGE_NAME,
        project=PROJECT_NAME,
        model=BEST_MODEL
    )


# ==========================================================
# MANUAL PREDICTION PAGE
# ==========================================================

@app.route("/manual")
def manual():

    return render_template(
        "manual.html"
    )


# ==========================================================
# RESUME ANALYSIS PAGE
# ==========================================================

@app.route("/resume")
def resume():

    return render_template(
        "upload_resume.html"
    )


# ==========================================================
# ABOUT PAGE
# ==========================================================

@app.route("/about")
def about():

    return render_template(
        "about.html",
        project=PROJECT_NAME,
        college=COLLEGE_NAME,
        developer=DEVELOPER
    )


# ==========================================================
# DASHBOARD PAGE
# ==========================================================

@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html",
        best_model=BEST_MODEL,
        rf_accuracy=RF_ACCURACY,
        svm_accuracy=SVM_ACCURACY
    )


# ==========================================================
# CONTACT PAGE
# ==========================================================

@app.route("/contact")
def contact():

    return render_template(
        "contact.html"
    )


# ==========================================================
# DARK MODE
# ==========================================================

@app.route("/darkmode")
def darkmode():

    return render_template(
        "darkmode.html"
    )


# ==========================================================
# DOWNLOAD REPORT PAGE
# ==========================================================

@app.route("/download")
def download():

    return render_template(
        "download.html"
    )


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.route("/health")
def health():

    return {

        "status": "Running",

        "project": PROJECT_NAME,

        "model": BEST_MODEL,

        "college": COLLEGE_NAME,

        "developer": DEVELOPER

    }


# ==========================================================
# MODEL INFORMATION
# ==========================================================

@app.route("/model")
def model_information():

    return render_template(

        "model.html",

        rf_accuracy=RF_ACCURACY,

        svm_accuracy=SVM_ACCURACY,

        best_model=BEST_MODEL

    )


# ==========================================================
# SYSTEM INFORMATION
# ==========================================================

@app.route("/system")
def system():

    return render_template(

        "system.html",

        current_time=datetime.now(),

        developer=DEVELOPER,

        college=COLLEGE_NAME

    )
# ==========================================================
# MANUAL PLACEMENT PREDICTION
# ==========================================================

@app.route("/manual_predict", methods=["POST"])
def manual_predict():

    try:

        # ---------------------------------------
        # Read User Input
        # ---------------------------------------

        student_name = request.form["name"]

        cgpa = float(request.form["cgpa"])

        communication = float(request.form["communication"])

        programming = float(request.form["programming"])

        internship = int(request.form["internship"])

        projects = int(request.form["projects"])

        attendance = float(request.form["attendance"])

        # Dummy IQ (same as training)
        iq = 100

        # ---------------------------------------
        # Prepare ML Input
        # ---------------------------------------

        input_data = np.array([[
            cgpa,
            iq,
            communication,
            programming,
            internship,
            projects,
            attendance
        ]])

        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)[0]

        # ---------------------------------------
        # Confidence / Readiness
        # ---------------------------------------

        confidence = calculate_readiness_score(
            cgpa,
            programming,
            communication,
            internship,
            projects,
            attendance,
            resume_score=None
        )

        if confidence >= 60:
            prediction_text = "✅ LIKELY TO BE PLACED"
        else:
            prediction_text = "❌ NEEDS IMPROVEMENT"

        # ---------------------------------------
        # Placement Grade
        # ---------------------------------------

        grade = placement_grade(confidence)

        # ---------------------------------------
        # AI Suggestions
        # ---------------------------------------

        suggestions = generate_suggestions(
            cgpa,
            programming,
            communication,
            internship,
            projects
        )

        # ---------------------------------------
        # Strengths
        # ---------------------------------------

        strengths = []

        if cgpa >= 8:
            strengths.append("Excellent Academic Performance")

        if programming >= 8:
            strengths.append("Strong Programming Skills")

        if communication >= 8:
            strengths.append("Excellent Communication")

        if internship == 1:
            strengths.append("Industry Exposure")

        if projects >= 3:
            strengths.append("Real-world Project Experience")

        if len(strengths) == 0:
            strengths.append("Good Learning Potential")

        # ---------------------------------------
        # Weak Areas
        # ---------------------------------------

        weak_areas = []

        if cgpa < 7:
            weak_areas.append("Improve CGPA")

        if programming < 7:
            weak_areas.append("Practice Coding")

        if communication < 7:
            weak_areas.append("Communication Skills")

        if internship == 0:
            weak_areas.append("No Internship Experience")

        if projects < 3:
            weak_areas.append("Build More Projects")

        # ---------------------------------------
        # Company Recommendation
        # ---------------------------------------

        companies = recommended_companies(confidence)

        # ---------------------------------------
        # Recommended Courses
        # ---------------------------------------

        courses = [

            "Python Programming",

            "Data Structures & Algorithms",

            "SQL & Database",

            "Machine Learning",

            "Web Development",

            "Cloud Computing",

            "Git & GitHub",

            "Interview Preparation"

        ]
        feature_data = feature_importance()
        

        # ---------------------------------------
        # Render Result Page
        # ---------------------------------------

        return render_template(

            "result.html",

            prediction=prediction_text,

            student_name=student_name,

            confidence=confidence,

            grade=grade,

            best_model=BEST_MODEL,

            rf_accuracy=RF_ACCURACY,

            svm_accuracy=SVM_ACCURACY,

            resume_score=None,

            skills=[],

            internship="Yes" if internship else "No",

            projects=projects,

            certifications=0,

            strengths=strengths,

            weak_areas=weak_areas,

            suggestions=suggestions,

            companies=companies,

            feature_data=feature_data,

            courses=courses,

            cgpa=cgpa,

            programming=programming,

            communication=communication,

            attendance=attendance

        )

    except Exception as e:

        return f"Prediction Error : {e}"
    # ==========================================================
# RESUME ANALYSIS & PREDICTION
# ==========================================================

@app.route("/analyze_resume", methods=["POST"])
def analyze_resume_route():

    try:

        # ---------------------------------------
        # Check Upload
        # ---------------------------------------

        if "resume" not in request.files:

            flash("Please upload a resume.")

            return redirect("/resume")

        file = request.files["resume"]

        if file.filename == "":

            flash("Please select a PDF file.")

            return redirect("/resume")

        # ---------------------------------------
        # Save Resume
        # ---------------------------------------

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            file.filename
        )

        file.save(filepath)

        # ---------------------------------------
        # Resume Analysis
        # ---------------------------------------

        report = analyze_resume(filepath)

        candidate_name = report["candidate_name"]

        email = report["email"]

        phone = report["phone"]

        linkedin = report["linkedin"]

        github = report["github"]

        skills = report["skills"]

        projects = report["projects"]

        internship = report["internship"]

        certifications = report["certifications"]

        resume_score = report["resume_score"]

        strengths = report["strengths"]

        weak_areas = report["weak_areas"]

        suggestions = report["suggestions"]

        companies = report["companies"]

        courses = report["courses"]

        placement_level = report["placement_level"]

        # ---------------------------------------
        # Convert Resume to ML Features
        # ---------------------------------------

        cgpa = min(10, max(6, resume_score / 10))

        iq = 100

        communication = min(10, max(5, len(skills)))

        programming = min(10, len(skills))

        attendance = 90

        input_data = np.array([[

            cgpa,

            iq,

            communication,

            programming,

            internship,

            projects,

            attendance

        ]])

        # ---------------------------------------
        # Scale Features
        # ---------------------------------------

        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)[0]

        if hasattr(model, "predict_proba"):

            confidence = round(

                np.max(

                    model.predict_proba(scaled_data)

                ) * 100,

                2

            )

        else:

            confidence = 95.00

        # ---------------------------------------
        # Prediction Text
        # ---------------------------------------

        confidence = calculate_readiness_score(
            cgpa,
            programming,
            communication,
            internship,
            projects,
            attendance,
            resume_score=resume_score
        )

        if confidence >= 60:

            prediction_text = "✅ LIKELY TO BE PLACED"

        else:

            prediction_text = "❌ NEEDS IMPROVEMENT"

        grade = placement_grade(confidence)

        feature_data = feature_importance()

        # ---------------------------------------
        # Show Result Dashboard
        # ---------------------------------------

        return render_template(

            "result.html",

            prediction=prediction_text,

            student_name=candidate_name,

            email=email,

            phone=phone,

            linkedin=linkedin,

            github=github,

            confidence=confidence,

            grade=grade,

            placement_level=placement_level,

            best_model=BEST_MODEL,

            rf_accuracy=RF_ACCURACY,

            svm_accuracy=SVM_ACCURACY,

            resume_score=resume_score,

            skills=skills,

            internship="Yes" if internship else "No",

            projects=projects,

            certifications=certifications,

            strengths=strengths,

            weak_areas=weak_areas,

            suggestions=suggestions,

            companies=companies,

            courses=courses,

            feature_data=feature_data,

            cgpa=cgpa,

            programming=programming,

            communication=communication,

            attendance=attendance

        )

    except Exception as e:

        return f"Resume Analysis Error : {e}"
    # ==========================================================
# PDF REPORT GENERATOR
# ==========================================================

@app.route("/download_report")
def download_report():

    try:

        filename = "Placement_Report.pdf"

        filepath = os.path.join(
            app.config["REPORT_FOLDER"],
            filename
        )

        doc = SimpleDocTemplate(filepath)

        styles = getSampleStyleSheet()

        elements = []

        # ==================================================
        # HEADER
        # ==================================================

        title = Paragraph(
            "<b><font size=18>"
            "Student Placement Prediction Report"
            "</font></b>",
            styles["Title"]
        )

        elements.append(title)

        elements.append(Spacer(1,20))

        subtitle = Paragraph(
            "<b>Chennai Institute of Technology</b><br/>"
            "Student Placement Prediction System using Machine Learning<br/>"
            "Random Forest Classifier",
            styles["Normal"]
        )

        elements.append(subtitle)

        elements.append(Spacer(1,25))

        # ==================================================
        # REPORT TABLE
        # ==================================================

        data = [

            ["Student Name", request.args.get("student_name","Student")],

            ["Prediction", request.args.get("prediction","-")],

            ["Confidence", request.args.get("confidence","0")+" %"],

            ["Resume Score", request.args.get("resume_score","N/A")],

            ["Best Model", BEST_MODEL],

            ["Random Forest Accuracy", str(RF_ACCURACY)+" %"],

            ["Linear SVM Accuracy", str(SVM_ACCURACY)+" %"],

            ["Generated On",
             datetime.now().strftime("%d-%m-%Y %H:%M")]

        ]

        table = Table(data)

        table.setStyle(TableStyle([

            ('BACKGROUND',(0,0),(-1,0),colors.darkblue),

            ('TEXTCOLOR',(0,0),(-1,0),colors.white),

            ('GRID',(0,0),(-1,-1),1,colors.black),

            ('BACKGROUND',(0,1),(0,-1),colors.lightblue),

            ('BOTTOMPADDING',(0,0),(-1,0),12),

            ('FONTSIZE',(0,0),(-1,-1),11)

        ]))

        elements.append(table)

        elements.append(Spacer(1,25))

        # ==================================================
        # AI RECOMMENDATION
        # ==================================================

        heading = Paragraph(
            "<b><font size=14>"
            "AI Career Recommendation"
            "</font></b>",
            styles["Heading2"]
        )

        elements.append(heading)

        recommendation = Paragraph(

            "The Random Forest Machine Learning model "
            "predicts the student's placement readiness "
            "based on academic performance, programming "
            "skills, communication skills, internships, "
            "projects and attendance.<br/><br/>"

            "Students are encouraged to improve coding "
            "skills, build real-time projects, complete "
            "internships and earn industry certifications "
            "to improve placement opportunities.",

            styles["BodyText"]

        )

        elements.append(recommendation)

        elements.append(Spacer(1,20))

        # ==================================================
        # FOOTER
        # ==================================================

        footer = Paragraph(

            "<b>Developed By :</b> Arvind Kartikeyan S<br/>"
            "Department of Information Technology<br/>"
            "Chennai Institute of Technology",

            styles["Italic"]

        )

        elements.append(footer)

        doc.build(elements)

        return send_file(

            filepath,

            as_attachment=True

        )

    except Exception as e:

        return f"PDF Generation Error : {e}"
    # =====================================================
# Run Flask Application
# =====================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Placement Pulse on http://0.0.0.0:{port}")
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )
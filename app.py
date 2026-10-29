import os
import pickle
from flask import Flask, render_template_string, request
import numpy as np

app = Flask(__name__)

# Load the trained AdaBoost model
MODEL_PATH = "Adaboost.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Warning: Could not load model file '{MODEL_PATH}': {e}")

# Modern Dark Theme HTML Template with responsive grid and card styling
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Admission & Success Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: #111827;
            --border-color: #1f2937;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent-glow: rgba(99, 102, 241, 0.4);
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --success: #10b981;
            --danger: #ef4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px 20px;
        }

        .container {
            width: 100%;
            max-width: 950px;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.6), 0 10px 10px -5px rgba(0, 0, 0, 0.4);
            padding: 40px;
            margin-bottom: 30px;
        }

        header {
            text-align: center;
            margin-bottom: 35px;
        }

        header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }

        header p {
            color: var(--text-secondary);
            font-size: 1rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 20px;
        }

        .input-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 0.82rem;
            font-weight: 500;
            color: var(--text-secondary);
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        input, select {
            background-color: #030712;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px 16px;
            color: var(--text-primary);
            font-size: 0.95rem;
            transition: all 0.3s ease;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }

        .btn-submit {
            grid-column: 1 / -1;
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 14px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 4px 14px var(--accent-glow);
            margin-top: 15px;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
        }

        .result-card {
            margin-top: 30px;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            animation: fadeIn 0.4s ease-in-out;
            border: 1px solid var(--border-color);
            background: #030712;
        }

        .result-card.success {
            border-color: rgba(16, 185, 129, 0.4);
            box-shadow: 0 0 25px rgba(16, 185, 129, 0.15);
        }

        .result-card.danger {
            border-color: rgba(239, 68, 68, 0.4);
            box-shadow: 0 0 25px rgba(239, 68, 68, 0.15);
        }

        .result-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }

        .result-value {
            font-size: 1.6rem;
            font-weight: 700;
        }

        .success .result-value { color: var(--success); }
        .danger .result-value { color: var(--danger); }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        footer {
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.85rem;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Student Admission & Success Portal</h1>
            <p>Evaluate candidate profiles powered by your AdaBoost Classifier model</p>
        </header>

        <form method="POST">
            <div class="form-grid">
                <div class="input-group">
                    <label>Age</label>
                    <input type="number" name="age" required value="{{ form_data.get('age', '18') }}">
                </div>
                <div class="input-group">
                    <label>Gender</label>
                    <select name="gender">
                        <option value="0" {% if form_data.get('gender') == '0' %}selected{% endif %}>Female / Other</option>
                        <option value="1" {% if form_data.get('gender') == '1' %}selected{% endif %}>Male</option>
                    </select>
                </div>
                <div class="input-group">
                    <label>State Code</label>
                    <input type="number" name="state" required value="{{ form_data.get('state', '1') }}">
                </div>
                <div class="input-group">
                    <label>Family Income ($)</label>
                    <input type="number" name="family_income" required value="{{ form_data.get('family_income', '75000') }}">
                </div>
                <div class="input-group">
                    <label>High School GPA</label>
                    <input type="number" step="0.01" name="high_school_gpa" required value="{{ form_data.get('high_school_gpa', '3.5') }}">
                </div>
                <div class="input-group">
                    <label>SAT Score</label>
                    <input type="number" name="sat_score" required value="{{ form_data.get('sat_score', '1250') }}">
                </div>
                <div class="input-group">
                    <label>ACT Score</label>
                    <input type="number" name="act_score" required value="{{ form_data.get('act_score', '28') }}">
                </div>
                <div class="input-group">
                    <label>Attendance Rate (%)</label>
                    <input type="number" step="0.1" name="attendance_rate" required value="{{ form_data.get('attendance_rate', '95.0') }}">
                </div>
                <div class="input-group">
                    <label>AP Courses Taken</label>
                    <input type="number" name="ap_courses" required value="{{ form_data.get('ap_courses', '4') }}">
                </div>
                <div class="input-group">
                    <label>Extracurricular Count</label>
                    <input type="number" name="extracurricular_count" required value="{{ form_data.get('extracurricular_count', '3') }}">
                </div>
                <div class="input-group">
                    <label>Volunteer Hours</label>
                    <input type="number" name="volunteer_hours" required value="{{ form_data.get('volunteer_hours', '50') }}">
                </div>
                <div class="input-group">
                    <label>Leadership Positions</label>
                    <input type="number" name="leadership_positions" required value="{{ form_data.get('leadership_positions', '2') }}">
                </div>
                <div class="input-group">
                    <label>Coding Projects</label>
                    <input type="number" name="coding_projects" required value="{{ form_data.get('coding_projects', '2') }}">
                </div>
                <div class="input-group">
                    <label>Social Media Hours / Week</label>
                    <input type="number" step="0.1" name="social_media_hours" required value="{{ form_data.get('social_media_hours', '10') }}">
                </div>
                <div class="input-group">
                    <label>Online Certifications</label>
                    <input type="number" name="online_certifications" required value="{{ form_data.get('online_certifications', '3') }}">
                </div>
                <div class="input-group">
                    <label>Essay Score (0-100)</label>
                    <input type="number" step="0.1" name="essay_score" required value="{{ form_data.get('essay_score', '85') }}">
                </div>
                <div class="input-group">
                    <label>Recommendation Score (0-100)</label>
                    <input type="number" step="0.1" name="recommendation_score" required value="{{ form_data.get('recommendation_score', '90') }}">
                </div>
                <div class="input-group">
                    <label>Interview Score (0-100)</label>
                    <input type="number" step="0.1" name="interview_score" required value="{{ form_data.get('interview_score', '88') }}">
                </div>

                <button type="submit" class="btn-submit">Run Prediction Analysis</button>
            </div>
        </form>

        {% if prediction is not none %}
        <div class="result-card {% if prediction == 1 %}success{% else %}danger{% endif %}">
            <div class="result-title">Model Evaluation Output</div>
            <div class="result-value">
                {% if prediction == 1 %}
                    Positive Admission Outcome / High Success Probability 🎉
                {% else %}
                    Review Recommended / Lower Success Probability ⚠️
                {% endif %}
            </div>
        </div>
        {% endif %}
    </div>

    <footer>
        Deployed with Flask & Render &bull; AdaBoost Machine Learning App
    </footer>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    form_data = {}

    if request.method == "POST":
        form_data = request.form
        try:
            features = [
                float(form_data.get("age", 18)),
                float(form_data.get("gender", 0)),
                float(form_data.get("state", 1)),
                float(form_data.get("family_income", 75000)),
                float(form_data.get("high_school_gpa", 3.5)),
                float(form_data.get("sat_score", 1250)),
                float(form_data.get("act_score", 28)),
                float(form_data.get("attendance_rate", 95.0)),
                float(form_data.get("ap_courses", 4)),
                float(form_data.get("extracurricular_count", 3)),
                float(form_data.get("volunteer_hours", 50)),
                float(form_data.get("leadership_positions", 2)),
                float(form_data.get("coding_projects", 2)),
                float(form_data.get("social_media_hours", 10)),
                float(form_data.get("online_certifications", 3)),
                float(form_data.get("essay_score", 85)),
                float(form_data.get("recommendation_score", 90)),
                float(form_data.get("interview_score", 88)),
            ]

            if model is not None:
                final_features = np.array([features])
                prediction = int(model.predict(final_features)[0])
            else:
                prediction = 1
        except Exception as e:
            print(f"Prediction error: {e}")
            prediction = None

    return render_template_string(
        TEMPLATE, prediction=prediction, form_data=form_data
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
import joblib


MODEL_PATH = "model/employment_model.pkl"

model = joblib.load(MODEL_PATH)


def get_skill_gaps(
    attendance,
    assessment,
    practical,
    completion,
):
    gaps = []

    if attendance < 75:
        gaps.append("Low Attendance")

    if assessment < 60:
        gaps.append("Assessment Performance")

    if practical < 60:
        gaps.append("Practical Industry Skills")

    if completion < 80:
        gaps.append("Course Completion")

    if not gaps:
        gaps.append("No Major Skill Gap Detected")

    return gaps


def get_recommendations(gaps):
    recommendations = []

    for gap in gaps:

        if gap == "Low Attendance":
            recommendations.append(
                "Attendance intervention and trainee follow-up"
            )

        elif gap == "Assessment Performance":
            recommendations.append(
                "Revision support and assessment preparation"
            )

        elif gap == "Practical Industry Skills":
            recommendations.append(
                "Advanced practical training and industry project"
            )

        elif gap == "Course Completion":
            recommendations.append(
                "Complete pending training modules"
            )

        elif gap == "No Major Skill Gap Detected":
            recommendations.append(
                "Employer referral and placement support"
            )

    return recommendations


def predict_employment(
    attendance,
    assessment,
    practical,
    completion,
    experience,
):

    features = [[
        attendance,
        assessment,
        practical,
        completion,
        experience,
    ]]

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    employment_probability = probabilities[1] * 100

    if employment_probability >= 75:
        risk = "Low"
    elif employment_probability >= 50:
        risk = "Medium"
    else:
        risk = "High"

    outcome = (
        "Likely Employed"
        if prediction == 1
        else "Needs Intervention"
    )

    gaps = get_skill_gaps(
        attendance,
        assessment,
        practical,
        completion,
    )

    recommendations = get_recommendations(gaps)

    return {
        "employment_probability": round(
            employment_probability, 2
        ),
        "outcome": outcome,
        "risk": risk,
        "skill_gaps": gaps,
        "recommendations": recommendations,
    }
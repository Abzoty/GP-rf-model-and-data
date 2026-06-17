"""
03_interactive_ui.py

Streamlit UI for predicting the most likely specialization/program from a JSON
registration record. Supports the expanded per-course schema:
- grade
- points
- termWork
- examWork
- result
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Student Program Predictor", page_icon="🎓", layout="wide")

# --- PATHS ---
ARTIFACT_DIR = Path("artifacts")
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "Best_Calibrated_Model.pkl"
MODEL_INFO_PATH = ARTIFACT_DIR / "best_model_info.json"

# --- GRADE MAPPINGS ---
GRADE_ORDER = {
    "Not_Registered": 0,
    "F": 1,
    "Abs": 2,
    "Con": 3,
    "D": 4,
    "D+": 5,
    "C": 6,
    "C+": 7,
    "P": 8,
    "B": 9,
    "B+": 10,
    "A": 11,
    "A+": 12,
}

GRADE_TO_POINTS = {
    "Not_Registered": 0.0,
    "F": 0.0,
    "Abs": 0.0,
    "Con": 0.0,
    "D": 2.0,
    "D+": 2.2,
    "C": 2.4,
    "C+": 2.7,
    "P": 0.0,
    "B": 3.0,
    "B+": 3.3,
    "A": 3.7,
    "A+": 4.0,
}


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def normalize_grade(value) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan"}:
        return "Not_Registered"
    return text


def build_course_index(raw_json_data: list[dict]) -> dict[str, dict]:
    course_index = {}
    for record in raw_json_data:
        course = record.get("course", {}) or {}
        code = str(course.get("code", "")).strip()
        if code:
            course_index[code] = record
    return course_index


@st.cache_resource
def load_assets():
    """Load encoders, feature list, and the trained model."""
    try:
        label_enc = joblib.load(ARTIFACT_DIR / "label_encoder.pkl")
        selected_features = joblib.load(ARTIFACT_DIR / "selected_features.pkl")
        model = joblib.load(MODEL_PATH)

        model_info = {}
        if MODEL_INFO_PATH.exists():
            with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
                model_info = json.load(f)

        return label_enc, selected_features, model, model_info
    except Exception as e:
        st.error(f"Error loading assets: {e}. Run Scripts 1 and 2 first.")
        st.stop()


def parse_uploaded_json(uploaded_file) -> list[dict] | None:
    try:
        raw = json.load(uploaded_file)
    except Exception as e:
        st.error(f"Invalid JSON file: {e}")
        return None
    return raw


def parse_text_json(json_text: str) -> list[dict] | None:
    try:
        raw = json.loads(json_text)
    except Exception as e:
        st.error(f"Invalid JSON format: {e}")
        return None
    return raw


def extract_input_vector(raw_json_data: list[dict], selected_features: list[str]):
    input_data = {feature: 0.0 for feature in selected_features}

    course_index = build_course_index(raw_json_data)
    first_record = raw_json_data[0] if raw_json_data else {}
    student_node = first_record.get("student", {}) or {}
    student_info = {str(k).strip().lower(): v for k, v in student_node.items()}

    found_courses: list[str] = []

    for feature in selected_features:
        clean_feature = feature.strip()

        if clean_feature.endswith("_grade"):
            course_code = clean_feature[: -len("_grade")].strip()
            reg = course_index.get(course_code)
            if reg is not None:
                grade = normalize_grade(reg.get("grade"))
                input_data[clean_feature] = float(GRADE_ORDER.get(grade, 0))
                found_courses.append(course_code)

        elif clean_feature.endswith("_points"):
            course_code = clean_feature[: -len("_points")].strip()
            reg = course_index.get(course_code)
            if reg is not None:
                points = reg.get("points")
                if points is None:
                    grade = normalize_grade(reg.get("grade"))
                    points = GRADE_TO_POINTS.get(grade, 0.0)
                input_data[clean_feature] = safe_float(points, 0.0)

        elif clean_feature.endswith("_termWork"):
            course_code = clean_feature[: -len("_termWork")].strip()
            reg = course_index.get(course_code)
            if reg is not None:
                input_data[clean_feature] = safe_float(reg.get("termWork"), 0.0)

        elif clean_feature.endswith("_examWork"):
            course_code = clean_feature[: -len("_examWork")].strip()
            reg = course_index.get(course_code)
            if reg is not None:
                input_data[clean_feature] = safe_float(reg.get("examWork"), 0.0)

        elif clean_feature.endswith("_result"):
            course_code = clean_feature[: -len("_result")].strip()
            reg = course_index.get(course_code)
            if reg is not None:
                result = reg.get("result")
                if result is None:
                    result = safe_float(reg.get("termWork"), 0.0) + safe_float(reg.get("examWork"), 0.0)
                input_data[clean_feature] = safe_float(result, 0.0)

        elif clean_feature.endswith("_registered"):
            course_code = clean_feature[: -len("_registered")].strip()
            if course_code in course_index:
                input_data[clean_feature] = 1.0

        else:
            val = student_info.get(clean_feature.lower())
            if val is not None:
                input_data[clean_feature] = safe_float(val, 0.0)

    return input_data, found_courses, student_node


# --- MAIN UI ---
st.title("🎓 Student Program AI Profiler")
st.markdown(
    "Upload or paste a student's JSON academic record to generate a probability "
    "profile across the available departments."
)

label_enc, selected_features, model, model_info = load_assets()

with st.sidebar:
    st.header("⚙️ AI Engine Status")
    st.success(f"✅ Active Engine: {model_info.get('best_model_name', 'Calibrated Tree Model')}")
    st.markdown("---")
    st.info(
        "This app uses the exact same feature rules as the training pipeline, "
        "including per-course grade, points, term work, exam work, and total result."
    )
    if model_info:
        st.caption(f"Selected features: {model_info.get('selected_features_count', len(selected_features))}")

st.subheader("📝 Student Data Input (JSON)")
tab1, tab2 = st.tabs(["📄 Upload JSON File", "✍️ Paste JSON Data"])

raw_json_data = None

with tab1:
    uploaded_file = st.file_uploader("Upload Student Registration JSON", type=["json"])
    if uploaded_file is not None:
        raw_json_data = parse_uploaded_json(uploaded_file)

with tab2:
    json_text = st.text_area("Paste JSON array here", height=220)
    if json_text.strip():
        raw_json_data = parse_text_json(json_text)

if raw_json_data:
    if not isinstance(raw_json_data, list) or len(raw_json_data) == 0:
        st.error("JSON must be a non-empty array of registration objects.")
    else:
        input_data, found_courses, student_node = extract_input_vector(raw_json_data, selected_features)

        st.success("✅ JSON successfully parsed and mapped to model features.")

        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Student ID", student_node.get("id", "Unknown"))
        col_info2.metric("GPA", student_node.get("gpa", "N/A"))
        col_info3.metric("Courses Matched", len(set(found_courses)))

        with st.expander("🔍 View Extracted Features"):
            feature_df = pd.DataFrame([input_data]).T.rename(columns={0: "Value"})
            st.dataframe(feature_df, use_container_width=True)

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button(
                "🚀 Analyze Student & Predict Top Programs",
                use_container_width=True,
                type="primary",
            )

        if predict_button:
            input_df = pd.DataFrame([input_data])[selected_features]

            with st.spinner("Calculating department probabilities..."):
                probabilities = model.predict_proba(input_df)[0]

            results = pd.DataFrame(
                {
                    "Program": label_enc.classes_,
                    "Probability": probabilities,
                }
            ).sort_values(by="Probability", ascending=False)

            top_program = results.iloc[0]["Program"]
            top_prob = results.iloc[0]["Probability"] * 100

            st.header("🎯 Probability Ranking Results")
            st.metric(
                label="Most Likely Department",
                value=top_program,
                delta=f"{top_prob:.1f}% confidence",
                delta_color="normal",
            )

            chart_col, data_col = st.columns([2, 1])

            with chart_col:
                fig = px.bar(
                    results,
                    x="Probability",
                    y="Program",
                    orientation="h",
                    text=results["Probability"].apply(lambda x: f"{x*100:.1f}%"),
                    color="Probability",
                    color_continuous_scale="Viridis",
                    title="Complete Probability Ranking",
                )
                fig.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    showlegend=False,
                    xaxis_range=[0, 1],
                )
                st.plotly_chart(fig, use_container_width=True)

            with data_col:
                st.subheader("Raw Confidence")
                for _, row in results.iterrows():
                    st.write(f"**{row['Program']}**: `{row['Probability']*100:.2f}%`")
                    st.progress(float(row["Probability"]))
else:
    st.info("Upload or paste JSON data to begin.")

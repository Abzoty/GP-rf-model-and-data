"""
03_interactive_ui_updated.py

Streamlit UI for predicting the most likely specialization/program from a JSON
registration record. Supports an interactive Department Selection Questionnaire
to capture student preferences and combines them with ML predictions.
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

# --- CONFIGURATION & WEIGHTS ---
# These control the fusion formula: final_score = (ALPHA * model) + (BETA * questionnaire)
ALPHA_MODEL = 0.5
BETA_QUESTIONNAIRE = 0.5

# --- PATHS ---
ARTIFACT_DIR = Path("artifacts")
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "Best_Calibrated_Model.pkl"
MODEL_INFO_PATH = ARTIFACT_DIR / "best_model_info.json"
QUESTIONNAIRE_PATH = Path("data/Questionnaire/CS_Department_Questionnaire.json")

# --- GRADE MAPPINGS ---
GRADE_ORDER = {
    "Not_Registered": 0, "F": 1, "Abs": 2, "Con": 3, "D": 4, "D+": 5, 
    "C": 6, "C+": 7, "P": 8, "B": 9, "B+": 10, "A": 11, "A+": 12,
}

GRADE_TO_POINTS = {
    "Not_Registered": 0.0, "F": 0.0, "Abs": 0.0, "Con": 0.0, "D": 2.0, 
    "D+": 2.2, "C": 2.4, "C+": 2.7, "P": 0.0, "B": 3.0, "B+": 3.3, 
    "A": 3.7, "A+": 4.0,
}

# --- INITIALIZE SESSION STATE ---
if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = -1
if "q_answers" not in st.session_state:
    st.session_state.q_answers = {}
if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = None

# --- HELPER FUNCTIONS ---
def safe_float(value, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except Exception:
        return default

def normalize_grade(value) -> str:
    text = str(value).strip()
    return "Not_Registered" if not text or text.lower() in {"none", "nan"} else text

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

@st.cache_data
def load_questionnaire():
    """Load the questionnaire JSON configuration."""
    try:
        with open(QUESTIONNAIRE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading questionnaire JSON: {e}")
        return None

def parse_uploaded_json(uploaded_file) -> list[dict] | None:
    try:
        return json.load(uploaded_file)
    except Exception as e:
        st.error(f"Invalid JSON file: {e}")
        return None

def parse_text_json(json_text: str) -> list[dict] | None:
    try:
        return json.loads(json_text)
    except Exception as e:
        st.error(f"Invalid JSON format: {e}")
        return None

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
            if reg := course_index.get(course_code):
                input_data[clean_feature] = float(GRADE_ORDER.get(normalize_grade(reg.get("grade")), 0))
                found_courses.append(course_code)
        elif clean_feature.endswith("_points"):
            course_code = clean_feature[: -len("_points")].strip()
            if reg := course_index.get(course_code):
                points = reg.get("points")
                if points is None:
                    points = GRADE_TO_POINTS.get(normalize_grade(reg.get("grade")), 0.0)
                input_data[clean_feature] = safe_float(points, 0.0)
        elif clean_feature.endswith("_termWork"):
            course_code = clean_feature[: -len("_termWork")].strip()
            if reg := course_index.get(course_code):
                input_data[clean_feature] = safe_float(reg.get("termWork"), 0.0)
        elif clean_feature.endswith("_examWork"):
            course_code = clean_feature[: -len("_examWork")].strip()
            if reg := course_index.get(course_code):
                input_data[clean_feature] = safe_float(reg.get("examWork"), 0.0)
        elif clean_feature.endswith("_result"):
            course_code = clean_feature[: -len("_result")].strip()
            if reg := course_index.get(course_code):
                result = reg.get("result")
                if result is None:
                    result = safe_float(reg.get("termWork"), 0.0) + safe_float(reg.get("examWork"), 0.0)
                input_data[clean_feature] = safe_float(result, 0.0)
        elif clean_feature.endswith("_registered"):
            course_code = clean_feature[: -len("_registered")].strip()
            if course_code in course_index:
                input_data[clean_feature] = 1.0
        else:
            if (val := student_info.get(clean_feature.lower())) is not None:
                input_data[clean_feature] = safe_float(val, 0.0)

    return input_data, found_courses, student_node

def standardize_dept_code(label: str, q_departments: list, q_mapping: dict) -> str:
    """Ensure model labels align with Questionnaire short codes (CS, IS, etc)."""
    label_upper = str(label).strip().upper()
    
    # Explicit mapping aliases to catch mismatches between Model classes and Questionnaire JSON
    aliases = {
        "OPERATION RESEARCH & DECISION SUPPORT": "DS",
        "OPERATIONS RESEARCH & DECISION SUPPORT": "DS",
        "COMPUTER SCIENCE": "CS",
        "INFORMATION TECHNOLOGY": "IT",
        "INFORMATION SYSTEMS": "IS",
        "ARTIFICIAL INTELLIGENCE": "AI"
    }
    
    if label_upper in aliases:
        return aliases[label_upper]
    
    if label_upper in [d.upper() for d in q_departments]:
        return label_upper
        
    for short_code, full_name in q_mapping.items():
        if full_name.upper() == label_upper:
            return short_code
            
    return label_upper

def calculate_questionnaire_probs(answers_dict: dict, q_data: dict) -> dict:
    """Scores user answers against the JSON rubric and returns normalized probabilities."""
    dept_codes = q_data["metadata"]["departments"]
    raw_scores = {dep: 0.0 for dep in dept_codes}

    for q in q_data["questions"]:
        q_id = q["id"]
        if q_id in answers_dict:
            selected_ans_id = answers_dict[q_id]
            # Find the answer dict
            ans_data = next((a for a in q["answers"] if a["id"] == selected_ans_id), None)
            if ans_data and "scores" in ans_data:
                for dep, score in ans_data["scores"].items():
                    raw_scores[dep] += score
    
    total_score = sum(raw_scores.values())
    if total_score > 0:
        return {dep: score / total_score for dep, score in raw_scores.items()}
    return {dep: 1.0 / len(dept_codes) for dep in dept_codes}

# --- MAIN UI ---
st.title("🎓 Student Program AI Profiler")
st.markdown("Upload a student's academic record, optionally complete the preference questionnaire, and generate a customized program recommendation.")

label_enc, selected_features, model, model_info = load_assets()
q_data = load_questionnaire()

with st.sidebar:
    st.header("⚙️ AI Engine Status")
    st.success(f"✅ Active Engine: {model_info.get('best_model_name', 'Calibrated Tree Model')}")
    st.markdown("---")
    st.info(f"Fusion Weights:\n\n**Academic (Model):** {ALPHA_MODEL*100}%\n**Preference (Q's):** {BETA_QUESTIONNAIRE*100}%")

st.subheader("1️⃣ Student Data Input (JSON)")
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
        
        # Display extracted features overview
        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Student ID", student_node.get("id", "Unknown"))
        col_info2.metric("GPA", student_node.get("gpa", "N/A"))
        col_info3.metric("Courses Matched", len(set(found_courses)))

        st.markdown("---")
        st.subheader("2️⃣ Analysis Configuration")
        
        # Interactive Layout Buttons
        action_col1, action_col2 = st.columns(2)
        
        with action_col1:
            st.info("💡 **Option A:** Fast path using only ML analysis of academic grades.")
            if st.button("🚀 Run Model-Only Analysis", use_container_width=True):
                st.session_state.analysis_mode = "model_only"
                st.session_state.wizard_step = -1

        with action_col2:
            st.info("🎯 **Option B:** Combine ML academic history with student preferences.")
            if st.button("📝 Start Questionnaire Wizard", type="primary", use_container_width=True):
                st.session_state.wizard_step = 0
                st.session_state.analysis_mode = "wizard"

        # --- QUESTIONNAIRE WIZARD UI ---
        if st.session_state.wizard_step >= 0 and q_data and st.session_state.analysis_mode == "wizard":
            st.markdown("---")
            
            # --- CSS FIXES FOR LARGER FONT AND MORE SPACING ---
            st.markdown(
                """
                <style>
                div.stRadio p {
                    font-size: 18px !important;
                }
                div.stRadio > div[role="radiogroup"] > div {
                    margin-bottom: 16px !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            
            total_questions = len(q_data["questions"])
            step = st.session_state.wizard_step

            if step < total_questions:
                # Show flashcard for current question
                current_q = q_data["questions"][step]
                
                with st.container():
                    st.markdown(f"### Question {step + 1} of {total_questions}")
                    st.progress((step) / total_questions)
                    st.write(f"**{current_q['text']}**")

                    # Setup radio options
                    options = [a["id"] for a in current_q["answers"]]
                    labels = {a["id"]: a["text"] for a in current_q["answers"]}
                    
                    # Persist previous selection if navigating backwards, otherwise default to None
                    default_index = options.index(st.session_state.q_answers[current_q["id"]]) if current_q["id"] in st.session_state.q_answers else None
                    
                    selected_val = st.radio(
                        "Select your preference:",
                        options=options,
                        format_func=lambda x: labels[x],
                        index=default_index,
                        key=f"radio_step_{step}"
                    )

                    # Navigation Buttons
                    st.write("")
                    nav_col1, nav_col2, _ = st.columns([1, 1, 4])
                    with nav_col1:
                        if st.button("⬅️ Back") and step > 0:
                            st.session_state.wizard_step -= 1
                            st.rerun()
                    with nav_col2:
                        # Disable "Next" until a radio option is actually selected
                        if st.button("Next ➡️", type="primary", disabled=(selected_val is None)):
                            st.session_state.q_answers[current_q["id"]] = selected_val
                            st.session_state.wizard_step += 1
                            st.rerun()
            else:
                st.success("🎉 Questionnaire Complete!")
                st.progress(1.0)
                st.markdown("Your preferences have been recorded. We are ready to combine them with your academic profile.")
                
                if st.button("🚀 Analyze Student + Questionnaire Data", type="primary", use_container_width=True):
                    st.session_state.analysis_mode = "combined"
                    st.rerun()

        # --- RESULTS & PREDICTION DISPLAY ---
        if st.session_state.analysis_mode in ["model_only", "combined"]:
            st.markdown("---")
            st.header("📊 Final AI Profiler Results")

            # 1. Run Model Logic
            input_df = pd.DataFrame([input_data])[selected_features]
            model_probs = model.predict_proba(input_df)[0]
            
            # Map Model outputs to short codes based on JSON dictionary mappings
            q_depts = q_data["metadata"]["departments"] if q_data else []
            q_map = q_data.get("department_codes", {}) if q_data else {}
            
            model_results = {}
            for cls_label, prob in zip(label_enc.classes_, model_probs):
                short_code = standardize_dept_code(cls_label, q_depts, q_map)
                model_results[short_code] = prob

            # 2. Branch Output (Model Only vs Combined)
            if st.session_state.analysis_mode == "model_only":
                st.warning("⚠️ Displaying Model-Only results. Preferences were ignored.")
                
                df_res = pd.DataFrame([{"Program": k, "Probability": v} for k, v in model_results.items()])
                df_res = df_res.sort_values(by="Probability", ascending=False)
                
                top_prog = df_res.iloc[0]["Program"]
                st.metric(label="Most Likely Department (Academic Only)", value=top_prog, delta=f"{df_res.iloc[0]['Probability']*100:.1f}% confidence")
                
                fig = px.bar(
                    df_res, x="Probability", y="Program", orientation="h",
                    text=df_res["Probability"].apply(lambda x: f"{x*100:.1f}%"),
                    color="Probability", title="Model Predictions"
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False, xaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

            elif st.session_state.analysis_mode == "combined":
                # Calculate Questionnaire logic
                q_probs = calculate_questionnaire_probs(st.session_state.q_answers, q_data)
                
                # Align ALL departments found in either dict
                all_departments = list(set(list(model_results.keys()) + list(q_probs.keys())))
                
                combined_records = []
                for dep in all_departments:
                    m_score = model_results.get(dep, 0.0)
                    q_score = q_probs.get(dep, 0.0)
                    c_score = (ALPHA_MODEL * m_score) + (BETA_QUESTIONNAIRE * q_score)
                    
                    combined_records.append({"Program": dep, "Score_Type": "1. Academic Model", "Probability": m_score})
                    combined_records.append({"Program": dep, "Score_Type": "2. Questionnaire", "Probability": q_score})
                    combined_records.append({"Program": dep, "Score_Type": "3. Combined Final", "Probability": c_score})

                df_combined = pd.DataFrame(combined_records)
                
                # Identify Winners
                df_final_only = df_combined[df_combined["Score_Type"] == "3. Combined Final"].sort_values(by="Probability", ascending=False)
                top_model = max(model_results, key=model_results.get)
                top_q = max(q_probs, key=q_probs.get)
                top_combined = df_final_only.iloc[0]["Program"]

                # Hero Metrics
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Top by Grades (Model)", top_model, f"{model_results[top_model]*100:.1f}%")
                m_col2.metric("Top by Preference (Q)", top_q, f"{q_probs[top_q]*100:.1f}%")
                m_col3.metric("🏆 Recommended Program", top_combined, f"{df_final_only.iloc[0]['Probability']*100:.1f}%")

                # Visuals: Grouped Bar Chart
                fig = px.bar(
                    df_combined,
                    x="Probability",
                    y="Program",
                    color="Score_Type",
                    barmode="group",
                    orientation="h",
                    title="Comprehensive Ranking Analysis",
                    color_discrete_map={
                        "1. Academic Model": "#636EFA",
                        "2. Questionnaire": "#00CC96",
                        "3. Combined Final": "#EF553B"
                    }
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_range=[0, 1], legend_title="Score Dimension")
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("🔍 View Raw Probability Tables"):
                    st.dataframe(df_combined.pivot(index="Program", columns="Score_Type", values="Probability").sort_values("3. Combined Final", ascending=False).style.format("{:.2%}"), use_container_width=True)
else:
    st.info("👈 Upload or paste JSON data to begin.")
import streamlit as st

import numpy
from models.predictionPipeline import PredictorPipeline
from models.modelClass import DistilBERTWithNumeric
from features.featureExtractor import FeatureExtractor
from LLM_model.llmModel import FeedbackMaker
from LLM_feedback.essay import Essay
from LLM_feedback.focus import Content, Grammar
from LLM_feedback.prompt_maker import FeedbackPrompt
from LLM_feedback.role import Teacher, Tutor, llmAssistant
from LLM_feedback.rules import CurrentRules
from LLM_feedback.score import Score
import torch
import pandas as pd
import random
from LLM_feedback.prompt_maker import create_feedback_prompt


device = "cpu"   # or "cuda" later

@st.cache_resource
def load_models():
    pipeline = PredictorPipeline(
        model_path="from_training/best_model_correct_larger_qwk.pth",
        scaler_path="from_training/scaler_larger_final.pkl",
        thresholds_path="from_training/thresholds.json",
        device="cpu"
    )

    llm = FeedbackMaker(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        device="cpu"
    )

    feature_extractor = FeatureExtractor()

    return pipeline, llm, feature_extractor

pipeline, llm, feature_extractor = load_models()

st.title("Autograder and feedback system for student essays")

ASAP = pd.read_csv("data/test_essays.csv")
ASAP = ASAP[ASAP["prompt_name"] != "Facial action coding system"]

assignments = ASAP["prompt_name"].unique().tolist()

selected_prompt = st.selectbox(
    "Select assignment",
    assignments
)

filtered = ASAP[ASAP["prompt_name"] == selected_prompt]

if filtered.empty:
    st.error("No data found for this prompt")
    st.stop()

row = filtered.iloc[0]

assignment_prompt = row["assignment"]
assignment_article = row["source_text_1"]

st.subheader("Assignment Prompt")
st.write(assignment_prompt)

st.subheader("Source Article")
st.write(assignment_article)

mode = st.radio(
    "Choose essay mode",
    ["Write your own essay", "Use random essay from dataset"]
)

essay_text = None

if mode == "Write your own essay":
    essay_text = st.text_area("Write your essay here")

elif mode == "Use random essay from dataset":
    key = f"random_essay_{selected_prompt}"

    if key not in st.session_state:
        st.session_state[key] = None

    if st.button("Get a random essay"):
        st.session_state[key] = filtered.sample(1).iloc[0]

    if st.session_state[key] is not None:
        random_row = st.session_state[key]

        st.subheader("Random Essay from Dataset")
        st.write(random_row["full_text"])

        essay_text = random_row["full_text"]

if st.button("Grade essay"):

    if not essay_text or not essay_text.strip():
        st.warning("No essay available")
        st.stop()

    features = feature_extractor.extract(
        text=essay_text,
        prompt=assignment_prompt
    )

    pooled_pred, final_score = pipeline.predict(
        text=essay_text,
        numeric_features=features
    )

    # Get true score if dataset mode
    key = f"random_essay_{selected_prompt}"
    true_score = None

    if mode == "Use random essay from dataset":
        if st.session_state.get(key) is not None:
            true_score = st.session_state[key]["score"]


    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Grade", int(final_score))

    with col2:
        if true_score is not None:
            st.metric("Actual Grade", int(true_score))

    essay_obj = Essay(
        essay=essay_text,
        assignment=assignment_prompt,
        assignment_article=assignment_article
    )

    st.subheader("Feedback")

    score_obj = Score(final_score)

    prompt_obj = create_feedback_prompt(
        score=score_obj,
        role_type="teacher",      # or "tutor", "assistant"
        outcome_type="content",   # or "grammar"
        essay=essay_obj
    )

    prompt = prompt_obj.generate_prompt()

    with st.spinner("Generating feedback..."):
        feedback = llm.make_feedback(prompt)

    st.write(feedback)

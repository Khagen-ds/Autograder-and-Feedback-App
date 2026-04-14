import streamlit
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

device = "cpu"   # or "cuda" later

feature_extractor = FeatureExtractor()

pipeline = PredictorPipeline(
    model_path="from_training/best_model_correct_larger_qwk.pth",
    scaler_path="from_training/scaler_larger_final.pkl",
    thresholds_path="from_training/thresholds.json",
    device=device
)


llm = FeedbackMaker(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    device=device
)
print("Pipeline loaded")
print("Feature extractor loaded")
print("LLM loaded")

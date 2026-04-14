import streamlit as st
import numpy as np

from models.predictionPipeline import PredictorPipeline
from models.modelClass import DistilBERTWithNumeric

from featureExtractor import FeatureExtractor
from LLM_feedback.essay import Essay
from LLM_feedback.focus import Content, Grammar
from LLM_feedback.prompt_maker import FeedbackPrompt
from LLM_feedback.role import Teacher, Tutor, LLM_assistant
from LLM_feedback.rules import Current_rules
from LLM_feedback.score import Score

from llmModel import FeedbackMaker



st.title("Autograder and feedback system for student essays")
st.write("Upload or write an essay to receive a score and feedback.")

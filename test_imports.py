print("SCRIPT STARTED")

import streamlit
print("streamlit OK")

import numpy
print("numpy OK")

from models.predictionPipeline import PredictorPipeline
print("pipeline OK")

from models.modelClass import DistilBERTWithNumeric
print("modelClass OK")

from featureExtractor import FeatureExtractor
print("featureExtractor OK")

from llmModel import FeedbackMaker
print("llmModel OK")

from LLM_feedback.essay import Essay
from LLM_feedback.focus import Content, Grammar
from LLM_feedback.prompt_maker import FeedbackPrompt
from LLM_feedback.role import Teacher, Tutor, LLM_assistant
from LLM_feedback.rules import Current_rules
from LLM_feedback.score import Score

print("LLM_feedback OK")

print("ALL IMPORTS SUCCESSFUL")

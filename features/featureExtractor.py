import re
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class FeatureExtractor:
    def __init__(self, prompt):
        self.prompt = prompt

        # Load models once (important for performance)
        self.nlp = spacy.load("en_core_web_sm")
        self.sim_model = SentenceTransformer("all-MiniLM-L6-v2")

    def count_words(self, text):
        return len(text.split())

    def count_unique_words(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        return len(set(words))

    def count_punctuation(self, text):
        return text.count(","), text.count(".")

    def part_of_speech(self, text):
        doc = self.nlp(text)
        noun_count = sum(1 for token in doc if token.pos_ == "NOUN")
        verb_count = sum(1 for token in doc if token.pos_ == "VERB")
        adj_count  = sum(1 for token in doc if token.pos_ == "ADJ")
        adv_count  = sum(1 for token in doc if token.pos_ == "ADV")

        return noun_count, verb_count, adj_count, adv_count

    def prompt_similarity(self, essay):
        essay_emb = self.sim_model.encode([essay])
        prompt_emb = self.sim_model.encode([self.prompt])
        return cosine_similarity(essay_emb, prompt_emb)[0][0]

    def extract(self, text):
        """
        Returns features in EXACT same order as training
        """

        word_count = self.count_words(text)
        unique_word_count = self.count_unique_words(text)

        comma_count, period_count = self.count_punctuation(text)

        noun_count, verb_count, adj_count, adv_count = self.part_of_speech(text)

        similarity = self.prompt_similarity(text)

        features = np.array([
            word_count,
            unique_word_count,
            comma_count,
            period_count,
            noun_count,
            verb_count,
            adj_count,
            adv_count,
            similarity
        ], dtype=np.float32)

        return features

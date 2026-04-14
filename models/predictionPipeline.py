import torch
import numpy as np
import json
import joblib
from transformers import DistilBertTokenizerFast
from models.modelClass import DistilBERTWithNumeric


class PredictorPipeline:
    def __init__(self, model_path: str, scaler_path: str, thresholds_path: str, device: str = "cpu"):
        self.device = torch.device(device)

        # tokenizer used during implementation and training
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased"
        )

        # scaler for consistency with what used during training
        self.scaler = joblib.load(scaler_path)

        # thresholds obtained during testing
        with open(thresholds_path, "r") as f:
            self.thresholds = np.sort(np.array(json.load(f)))

        # custom DistilBERT model (same as in training)
        numeric_feature_size = self.scaler.n_features_in_
        self.model = DistilBERTWithNumeric(
            numeric_feature_size=numeric_feature_size
        )

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))    #    Model trained on 9200+ essays
        self.model.to(self.device)
        self.model.eval()

    
    def create_chunks_single_essay(self, text, numeric_features, max_length=512):
        '''
        Function to create chungs for single essays. Modified from the function used during training. Goal: Convert long essays(input) into chunks of token lenght 512.
        Numeric features(input) are repeated for each chunk
        '''
        enc = self.tokenizer(text, add_special_tokens=True, return_tensors="pt")
        token_ids = enc["input_ids"].squeeze(0)

        # handle empty token/ essay
        if len(token_ids) == 0:
            return None, None, None

        input_ids_list = []
        attention_mask_list = []

        i = 0
        while i < len(token_ids):
            end = min(i + max_length, len(token_ids))
            chunk = token_ids[i:end]

            mask = torch.ones_like(chunk)

            pad_len = max_length - len(chunk)
            if pad_len > 0:
                chunk = torch.cat(
                    [chunk, torch.zeros(pad_len, dtype=torch.long)]
                )
                mask = torch.cat(
                    [mask, torch.zeros(pad_len, dtype=torch.long)]
                )

            input_ids_list.append(chunk)
            attention_mask_list.append(mask)

            if end == len(token_ids):
                break

            i += max_length  # stride removed

        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)

        # numeric features scaling
        numeric_features = self.scaler.transform(
            numeric_features.reshape(1, -1)
        )
        numeric_features = torch.tensor(
            numeric_features, dtype=torch.float
        )

        numeric_features = numeric_features.expand(
            input_ids.shape[0], -1
        )

        return input_ids, attention_mask, numeric_features

    
    def predict(self, text, numeric_features):
        '''
        Function to predict the grade of the provided essay(input)
        It preprocess the input, chunk the essay, run the trained model on each cunk. Then it aggregate the chunk predictions before convert it into a grade (regression)
        Inputs are the essay from the student + numeric features 
        '''
        self.model.eval()

        if text is None or len(text.strip()) == 0:
            return None, None

        input_ids, attention_mask, numeric_features = (
            self.create_chunks_single_essay(text, numeric_features)
        )

        if input_ids is None:
            return None, None

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        numeric_features = numeric_features.to(self.device)

        with torch.no_grad():
            preds = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numeric_features=numeric_features
            )

        # pool chunk predictions (0–5)
        pooled_pred = preds.mean().item()
        pooled_pred = np.clip(pooled_pred, 0, 5)

        # convert to score (0–5)
        score_0_5 = np.digitize(pooled_pred, self.thresholds)

        # correct output (1–6)
        final_score = score_0_5 + 1

        return pooled_pred, final_score

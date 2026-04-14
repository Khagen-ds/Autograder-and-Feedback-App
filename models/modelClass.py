import torch
import torch.nn as nn
from transformers import DistilBertModel


class DistilBERTWithNumeric(nn.Module):
    '''
    Custom DistilBERT to combine numeric features with text features. Predicts a continous score (regression)
    '''
    def __init__(self, numeric_feature_size, dropout=0.1):
        super(DistilBERTWithNumeric, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden_size = self.bert.config.hidden_size  #    Size of token embedding

        self.dropout = nn.Dropout(dropout)    #    prevents overfitting
        self.layercon1 = nn.Linear(hidden_size + numeric_feature_size, 256)    #    txt embed + num feature
        self.relu = nn.ReLU()        #    Non-linear
        self.layercon2 = nn.Linear(256, 1)    #    Final predicted essay score

    def forward(self, input_ids, attention_mask, numeric_features=None):
        #    input_ids = tokenized text
        #    attention_mask = tells models which tokens are padding
        #    numeric_features = additinal manually added features
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)  #    embeddings for every token
        mask = attention_mask.unsqueeze(-1)
        embeddings = outputs.last_hidden_state * mask
        sum_embeddings = embeddings.sum(dim=1)   
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)    #    prevent dividing by 0
        cls_embedding = sum_embeddings / sum_mask     #    mean pooling

        
        if numeric_features is not None:
            combined = torch.cat([cls_embedding, numeric_features], dim=1)    #    Combines / concatenates text features and numeric features
        else:
            combined = cls_embedding

        x = self.dropout(combined)
        x = self.layercon1(x)
        x = self.relu(x)
        x = self.dropout(x)
        regression_output = self.layercon2(x)
        return regression_output.squeeze(-1)     
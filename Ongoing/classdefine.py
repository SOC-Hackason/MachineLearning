import numpy as np
import torch 
from transformers import BertTokenizer, BertModel
import torch.nn as nn


class TopicClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TopicClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TopicClassifierCombined:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased').to(device)
        # load from topic_classifier.pth saved as state_dict
        self.classifier = TopicClassifier(input_dim=768, output_dim=2, hidden_dim=16).to(device)
        self.classifier.load_state_dict(torch.load("topic_classifier2.pth"))
        
        
    def predict(self, text):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
            outputs = self.model(**inputs)
            last_hidden_states = outputs.last_hidden_state[:, 0, :]
            logits = self.classifier(last_hidden_states)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            return probs.cpu().numpy()